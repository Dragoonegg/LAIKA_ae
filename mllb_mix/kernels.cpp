/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 *
 * Original work:
 *   Copyright (C) 2022–2024 Henrique Fingler
 *   Copyright (C) 2022–2024 Isha Tarte
 *
 * Modifications and adaptations for LAIKA:Machine Learning-Assisted In-Kernel APU Acceleration
 *   Copyright (C) 2024-2025 Haoming Zhuo
 *
 * This file is adapted from the original LAKE/MLLB kernel module.
 * Major changes include:
 *   - Integration with LAIKA framework
 *   - Hybrid execution support
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>
#include "consts.h"
#include <unistd.h>



__global__ void mllb_infer_v1(float* input, float* w1, float* b1,
    int n, int k, float* w2, float b2)
{
    __shared__ float sm[64];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < 10) {
        float acc = 0;

        for(int i = 0; i < 15; i++) {
            acc += input[i] * w1[i*k + id];
        }
        
        acc += b1[id];
        acc = acc > 0 ? acc : 0;
        sm[id] = acc;
    }

    __syncthreads();

    if (id == 0) {
        float res = 0;
        for(int i = 0; i < 10; i++) {
            res += sm[i] * w2[i];
        }
        input[0] = res+b2;
    }
}

__global__ __launch_bounds__(128) 
void mllb_infer_v2(float* inputs, 
    float* w1, float* b1, float* w2, float b2, float* results)
{   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 65536) {  // Support maximum batch_size
        float* input = inputs + tid * 15;  // Each input has 15 features
        
       
        float hidden[10] = {0};
        
        for (int j = 0; j < 10; j++) {
            float acc = 0;
            for (int i = 0; i < 15; i++) {
                acc += input[i] * w1[i*10 + j];
            }
            hidden[j] = acc + b1[j];
            hidden[j] = hidden[j] > 0 ? hidden[j] : 0;  // ReLU
        }
        float output = 0;
        for (int j = 0; j < 10; j++) {
            output += hidden[j] * w2[j];
        }
        output += b2;
        
        results[tid] = output;
    }
}




__global__ __launch_bounds__(256) void mllb_persistent_infer(
    float* inputs, float* w1, float* b1, float* w2, float b2, float* results,
    int* task_flag, int* quit_flag, int* batch_size)
{   
    while (true) {
        // Only perform memory barrier when needed
        __threadfence_system();
        if (*quit_flag) break;
        if (*task_flag == 1) {
            int bs = *batch_size;
            // Block/thread can be used to assign processing tasks here
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = tid; i < bs; i += gridDim.x * blockDim.x) {
                // Process the i-th input
                float* input = inputs + i * NR_FEAT;
                float acc[10] = {0};
                for (int j = 0; j < 10; ++j) {
                    for (int k = 0; k < NR_FEAT; ++k)
                        acc[j] += input[k] * w1[k*10 + j];
                    acc[j] += b1[j];
                    acc[j] = acc[j] > 0 ? acc[j] : 0;
                }
                float out = 0;
                for (int j = 0; j < 10; ++j)
                    out += acc[j] * w2[j];
                out += b2;
                results[i] = out;
            }
            __syncthreads();
            if (tid == 0) {
                *task_flag = 0; // Notify host that task is completed
                 // Ensure flag update is visible to host
            }
        }
        else{
            //__threadfence_system(); 
            __builtin_amdgcn_s_sleep(10000);        
        }
        // Reduce memory barrier frequency, only use when necessary
        
    }
}

// Can be adjusted as needed
#ifndef MIN_SLEEP_CYCLES
#define MIN_SLEEP_CYCLES    1024    // 1k cycles
#endif
#ifndef MAX_SLEEP_CYCLES
#define MAX_SLEEP_CYCLES    65536   // 65k cycles
#endif
#ifndef BACKOFF_FACTOR
#define BACKOFF_FACTOR      4
#endif
#ifndef IDLE_SPIN_TRIES
#define IDLE_SPIN_TRIES     8       // Do a few light-weight polls when idle to reduce wake-up latency
#endif

// Wait for incomplete memory transactions (equivalent to s_waitcnt vmcnt(0) etc.)
__device__ __forceinline__ void wait_vmem_all_complete() {
#if defined(__HIP_PLATFORM_AMD__)
    __builtin_amdgcn_s_waitcnt(0);
#endif
}

__global__ __launch_bounds__(256)
void mllb_persistent_infer_2(
    float* inputs, float* w1, float* b1, float* w2, float b2, float* results,
    int* task_flag, int* quit_flag, int* batch_size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared variables: only let one thread read global flags, then broadcast to reduce bus pressure
    __shared__ int s_quit;
    __shared__ int s_has_work;
    __shared__ int s_bs;

    int sleep_cycles = MIN_SLEEP_CYCLES;

    while (true) {
        // --- 1) Block leader reads flags and broadcasts ---
        if (threadIdx.x == 0) {
            wait_vmem_all_complete();        // Converge memory transactions before reading flags
            s_quit     = *quit_flag;
            s_has_work = (*task_flag == 1);
            s_bs       = s_has_work ? *batch_size : 0;  // Only fetch batch when there is a task
        }
        __syncthreads();

        if (s_quit) break;

        if (s_has_work) {
            // --- 2) Has task: reset backoff and execute computation ---
            sleep_cycles = MIN_SLEEP_CYCLES;

            const int bs = s_bs;
            for (int i = tid; i < bs; i += gridDim.x * blockDim.x) {
                float* input = inputs + i * NR_FEAT;

                // First layer: NR_FEAT x 10 + ReLU
                float acc[10];
#pragma unroll
                for (int j = 0; j < 10; ++j) acc[j] = 0.0f;

#pragma unroll 4
                for (int k = 0; k < NR_FEAT; ++k) {
                    float x = input[k];
#pragma unroll
                    for (int j = 0; j < 10; ++j) {
                        acc[j] += x * w1[k * 10 + j];
                    }
                }
#pragma unroll
                for (int j = 0; j < 10; ++j) {
                    float v = acc[j] + b1[j];
                    acc[j] = (v > 0.0f) ? v : 0.0f;
                }

                // Second layer: 10 -> 1
                float out = 0.0f;
#pragma unroll
                for (int j = 0; j < 10; ++j) out += acc[j] * w2[j];
                out += b2;

                results[i] = out;
            }

            __syncthreads();

            // Completion notification and visibility guarantee: only perform necessary system fences
            if (tid == 0) {
                __threadfence_system();   // First ensure results are visible
                *task_flag = 0;           // Then set to 0 to notify host of completion
                __threadfence_system();   // Optional: do once more to reinforce "write data before clearing flag"
            }
        } else {
            // --- 3) No task: short polling + s_sleep exponential backoff ---
            // To reduce perceived latency after host sets flag, quickly check a few times first
            bool observed = false;
            for (int t = 0; t < IDLE_SPIN_TRIES; ++t) {
                if (threadIdx.x == 0) {
                    wait_vmem_all_complete();
                    s_has_work = (*task_flag == 1);
                    if (s_has_work) s_bs = *batch_size;
                }
                __syncthreads();
                if (s_has_work) { observed = true; break; }
            }

            if (observed) {
                // Task found, immediately reset backoff and proceed to next round
                sleep_cycles = MIN_SLEEP_CYCLES;
                continue;
            }

            // Still no task: enter short sleep with exponential backoff
            wait_vmem_all_complete();
#if defined(__HIP_PLATFORM_AMD__)
            // Use loop to implement sleep, as __builtin_amdgcn_s_sleep requires constant parameter
            // Decompose sleep_cycles into multiple fixed-size sleep cycles
            int remaining = sleep_cycles;
            while (remaining >= 1024) {
                __builtin_amdgcn_s_sleep(1024);
                remaining -= 1024;
            }
            if (remaining > 0) {
                // For remaining cycles less than 1024, use a smaller fixed value
                __builtin_amdgcn_s_sleep(512);
            }
#else
            // Other platforms can replace with __nanosleep or idle loop (fallback solution)
#endif
            long next = (long)sleep_cycles * BACKOFF_FACTOR;
            sleep_cycles = (next > MAX_SLEEP_CYCLES) ? MAX_SLEEP_CYCLES : (int)next;
        }
        // Next round
    }
}



#ifndef WAVES_MIN
#define WAVES_MIN 1
#endif
#ifndef WAVES_MAX
#define WAVES_MAX 2
#endif

// Constrain registers/occupancy: 128 threads per block, at least 2 blocks can run concurrently
__launch_bounds__(256, 2)
__attribute__((amdgpu_waves_per_eu(WAVES_MIN, WAVES_MAX)))
__global__
void mllb_persistent_infer_2_opt(
    float* __restrict__ inputs,      // [bs * NR_FEAT]
    float* __restrict__ w1,          // [NR_FEAT * 10]
    float* __restrict__ b1,          // [10]
    float* __restrict__ w2,          // [10]
    float  b2,
    float* __restrict__ results,           // [bs]
    int* __restrict__ task_flag,
    int* __restrict__ quit_flag,
    int* __restrict__ batch_size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared state: leader polls and broadcasts
    __shared__ int s_quit, s_has_work, s_bs;

    // Copy b1/w2 to LDS (once per task), and pad to 16 length for vectorization convenience
    __shared__ float s_b1[16];
    __shared__ float s_w2[16];

    int sleep_cycles = MIN_SLEEP_CYCLES;

    while (true) {
        if (threadIdx.x == 0) {
            wait_vmem_all_complete();
            // Read with "acquire" semantics to prevent stale cache (volatile works well under HIP/Clang)
            s_quit = *quit_flag;
            int has = (*task_flag == 1);
            s_has_work = has;
            s_bs = has ? *batch_size : 0;
        }
        __syncthreads();
        if (s_quit) break;

        if (s_has_work) {
            sleep_cycles = MIN_SLEEP_CYCLES;

            // Load b1/w2 into LDS only once; set 10..15 to 0 to avoid branches
            if (threadIdx.x < 16) {
                s_b1[threadIdx.x] = (threadIdx.x < 10) ? b1[threadIdx.x] : 0.0f;
                s_w2[threadIdx.x] = (threadIdx.x < 10) ? w2[threadIdx.x] : 0.0f;
            }
            __syncthreads();

            const int bs = s_bs;
            const int stride = gridDim.x * blockDim.x;

            for (int i = tid; i < bs; i += stride) {
                const float* __restrict__ x = inputs + i * NR_FEAT;

                // 10-dimensional hidden layer accumulators, stored in registers
                float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
                float acc5 = 0, acc6 = 0, acc7 = 0, acc8 = 0, acc9 = 0;

                // k main loop: 4 features per group (float4)
                int k = 0;
#pragma unroll 2
                for (; k + 3 < NR_FEAT; k += 4) {
                    // Batch load 4 inputs (compiler will generate vector load under natural alignment)
                    float4 xv = *reinterpret_cast<const float4*>(x + k);
                    // Each feature corresponds to 10 weights: w1[(k+p)*10 + j] contiguous memory, cache-friendly
#pragma unroll
                    for (int j = 0; j < 10; ++j) {
                        float w0 = w1[(k + 0) * 10 + j];
                        float w1v = w1[(k + 1) * 10 + j];
                        float w2v = w1[(k + 2) * 10 + j];
                        float w3v = w1[(k + 3) * 10 + j];
                        // Unroll onto 10 acc registers; fmaf generates FMA
                        switch (j) {
                            case 0:  acc0 = fmaf(xv.x, w0, acc0); acc0 = fmaf(xv.y, w1v, acc0); acc0 = fmaf(xv.z, w2v, acc0); acc0 = fmaf(xv.w, w3v, acc0); break;
                            case 1:  acc1 = fmaf(xv.x, w0, acc1); acc1 = fmaf(xv.y, w1v, acc1); acc1 = fmaf(xv.z, w2v, acc1); acc1 = fmaf(xv.w, w3v, acc1); break;
                            case 2:  acc2 = fmaf(xv.x, w0, acc2); acc2 = fmaf(xv.y, w1v, acc2); acc2 = fmaf(xv.z, w2v, acc2); acc2 = fmaf(xv.w, w3v, acc2); break;
                            case 3:  acc3 = fmaf(xv.x, w0, acc3); acc3 = fmaf(xv.y, w1v, acc3); acc3 = fmaf(xv.z, w2v, acc3); acc3 = fmaf(xv.w, w3v, acc3); break;
                            case 4:  acc4 = fmaf(xv.x, w0, acc4); acc4 = fmaf(xv.y, w1v, acc4); acc4 = fmaf(xv.z, w2v, acc4); acc4 = fmaf(xv.w, w3v, acc4); break;
                            case 5:  acc5 = fmaf(xv.x, w0, acc5); acc5 = fmaf(xv.y, w1v, acc5); acc5 = fmaf(xv.z, w2v, acc5); acc5 = fmaf(xv.w, w3v, acc5); break;
                            case 6:  acc6 = fmaf(xv.x, w0, acc6); acc6 = fmaf(xv.y, w1v, acc6); acc6 = fmaf(xv.z, w2v, acc6); acc6 = fmaf(xv.w, w3v, acc6); break;
                            case 7:  acc7 = fmaf(xv.x, w0, acc7); acc7 = fmaf(xv.y, w1v, acc7); acc7 = fmaf(xv.z, w2v, acc7); acc7 = fmaf(xv.w, w3v, acc7); break;
                            case 8:  acc8 = fmaf(xv.x, w0, acc8); acc8 = fmaf(xv.y, w1v, acc8); acc8 = fmaf(xv.z, w2v, acc8); acc8 = fmaf(xv.w, w3v, acc8); break;
                            default: acc9 = fmaf(xv.x, w0, acc9); acc9 = fmaf(xv.y, w1v, acc9); acc9 = fmaf(xv.z, w2v, acc9); acc9 = fmaf(xv.w, w3v, acc9); break;
                        }
                    }
                }
                // Handle tail (0~3 features)
                for (; k < NR_FEAT; ++k) {
                    float xk = x[k];
                    acc0 = fmaf(xk, w1[k*10 + 0], acc0);
                    acc1 = fmaf(xk, w1[k*10 + 1], acc1);
                    acc2 = fmaf(xk, w1[k*10 + 2], acc2);
                    acc3 = fmaf(xk, w1[k*10 + 3], acc3);
                    acc4 = fmaf(xk, w1[k*10 + 4], acc4);
                    acc5 = fmaf(xk, w1[k*10 + 5], acc5);
                    acc6 = fmaf(xk, w1[k*10 + 6], acc6);
                    acc7 = fmaf(xk, w1[k*10 + 7], acc7);
                    acc8 = fmaf(xk, w1[k*10 + 8], acc8);
                    acc9 = fmaf(xk, w1[k*10 + 9], acc9);
                }

                // Add bias + ReLU (using b1 in LDS)
                acc0 = fmaxf(acc0 + s_b1[0], 0.0f);
                acc1 = fmaxf(acc1 + s_b1[1], 0.0f);
                acc2 = fmaxf(acc2 + s_b1[2], 0.0f);
                acc3 = fmaxf(acc3 + s_b1[3], 0.0f);
                acc4 = fmaxf(acc4 + s_b1[4], 0.0f);
                acc5 = fmaxf(acc5 + s_b1[5], 0.0f);
                acc6 = fmaxf(acc6 + s_b1[6], 0.0f);
                acc7 = fmaxf(acc7 + s_b1[7], 0.0f);
                acc8 = fmaxf(acc8 + s_b1[8], 0.0f);
                acc9 = fmaxf(acc9 + s_b1[9], 0.0f);

                // Second layer dot product (w2 in LDS)
                float out =
                    acc0 * s_w2[0] + acc1 * s_w2[1] + acc2 * s_w2[2] +
                    acc3 * s_w2[3] + acc4 * s_w2[4] + acc5 * s_w2[5] +
                    acc6 * s_w2[6] + acc7 * s_w2[7] + acc8 * s_w2[8] +
                    acc9 * s_w2[9] + b2;

                results[i] = out;
            }

            __syncthreads();
            if (tid == 0) {
                __threadfence_system();          // First ensure results are visible
                atomicExch((int*)task_flag, 0);  // Then clear flag (release)
                // Usually no need for second threadfence; Host side should use acquire after reading flag before reading results
            }
        } else {
            // No task: short polling + exponential backoff
            bool observed = false;
#pragma unroll
            for (int t = 0; t < IDLE_SPIN_TRIES; ++t) {
                if (threadIdx.x == 0) {
                    wait_vmem_all_complete();
                    int has = (*task_flag == 1);
                    s_has_work = has;
                    if (has) s_bs = *batch_size;
                }
                __syncthreads();
                if (s_has_work) { observed = true; break; }
            }
            if (observed) { sleep_cycles = MIN_SLEEP_CYCLES; continue; }

            wait_vmem_all_complete();
#if defined(__HIP_PLATFORM_AMD__)
            // Precise sleep: digest remaining cycles in segments
            int remaining = sleep_cycles;
            while (remaining >= 1024) { __builtin_amdgcn_s_sleep(1024); remaining -= 1024; }
            if (remaining >= 512)      { __builtin_amdgcn_s_sleep(512);  remaining -= 512;  }
            if (remaining >= 256)      { __builtin_amdgcn_s_sleep(256);  remaining -= 256;  }
            if (remaining > 0)         { __builtin_amdgcn_s_sleep(64); } // Finish up
#else
            // Other platforms: can replace with __nanosleep or idle loop
#endif
            long next = (long)sleep_cycles * BACKOFF_FACTOR;
            sleep_cycles = (next > MAX_SLEEP_CYCLES) ? MAX_SLEEP_CYCLES : (int)next;
        }
    }
}