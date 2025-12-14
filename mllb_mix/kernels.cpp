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
    
    if (tid < 65536) {  // 支持最大batch_size
        float* input = inputs + tid * 15;  // 每个输入15个特征
        
       
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
        // 只在需要时进行内存屏障
        __threadfence_system();
        if (*quit_flag) break;
        if (*task_flag == 1) {
            int bs = *batch_size;
            // 这里可以用 block/thread 分配处理任务
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = tid; i < bs; i += gridDim.x * blockDim.x) {
                // 处理第i个输入
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
                *task_flag = 0; // 通知host已完成
                 // 确保flag更新对host可见
            }
        }
        else{
            //__threadfence_system(); 
            __builtin_amdgcn_s_sleep(10000);        
        }
        // 减少内存屏障频率，只在必要时使用
        
    }
}

// 可按需调整
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
#define IDLE_SPIN_TRIES     8       // 空闲时先做几次轻量轮询以降低唤醒延迟
#endif

// 等待未完成的内存事务（等价于 s_waitcnt vmcnt(0) 等）
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

    // 共享变量：只让一个线程读全局标志，然后广播，减少总线压力
    __shared__ int s_quit;
    __shared__ int s_has_work;
    __shared__ int s_bs;

    int sleep_cycles = MIN_SLEEP_CYCLES;

    while (true) {
        // --- 1) 由块内 leader 读标志并广播 ---
        if (threadIdx.x == 0) {
            wait_vmem_all_complete();        // 收敛内存事务后再读标志
            s_quit     = *quit_flag;
            s_has_work = (*task_flag == 1);
            s_bs       = s_has_work ? *batch_size : 0;  // 仅有任务时才取 batch
        }
        __syncthreads();

        if (s_quit) break;

        if (s_has_work) {
            // --- 2) 有任务：重置退避并执行计算 ---
            sleep_cycles = MIN_SLEEP_CYCLES;

            const int bs = s_bs;
            for (int i = tid; i < bs; i += gridDim.x * blockDim.x) {
                float* input = inputs + i * NR_FEAT;

                // 第一层：NR_FEAT x 10 + ReLU
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

                // 第二层：10 -> 1
                float out = 0.0f;
#pragma unroll
                for (int j = 0; j < 10; ++j) out += acc[j] * w2[j];
                out += b2;

                results[i] = out;
            }

            __syncthreads();

            // 完成通知与可见性保障：只做必要的系统栅栏
            if (tid == 0) {
                __threadfence_system();   // 先确保 results 可见
                *task_flag = 0;           // 再置 0 通知 host 完成
                __threadfence_system();   // 可选：再做一次，巩固“先写数据后清旗”
            }
        } else {
            // --- 3) 无任务：短轮询 + s_sleep 指数退避 ---
            // 为了降低被 host 置位后的感知延迟，先快速检查几次
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
                // 发现有任务，立即重置退避，进入下一轮处理
                sleep_cycles = MIN_SLEEP_CYCLES;
                continue;
            }

            // 仍然没有任务：进入短睡眠并指数退避
            wait_vmem_all_complete();
#if defined(__HIP_PLATFORM_AMD__)
            // 使用循环实现睡眠，因为 __builtin_amdgcn_s_sleep 需要常量参数
            // 将 sleep_cycles 分解为多个固定大小的睡眠周期
            int remaining = sleep_cycles;
            while (remaining >= 1024) {
                __builtin_amdgcn_s_sleep(1024);
                remaining -= 1024;
            }
            if (remaining > 0) {
                // 对于剩余的小于1024的周期，使用一个较小的固定值
                __builtin_amdgcn_s_sleep(512);
            }
#else
            // 其他平台可替换为 __nanosleep 或空转（退化方案）
#endif
            long next = (long)sleep_cycles * BACKOFF_FACTOR;
            sleep_cycles = (next > MAX_SLEEP_CYCLES) ? MAX_SLEEP_CYCLES : (int)next;
        }
        // 下一轮
    }
}



#ifndef WAVES_MIN
#define WAVES_MIN 1
#endif
#ifndef WAVES_MAX
#define WAVES_MAX 2
#endif

// 约束寄存器/占用率：每块 128 线程，至少能并发 2 块
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

    // 共享状态：leader 轮询并广播
    __shared__ int s_quit, s_has_work, s_bs;

    // 将 b1/w2 拷到 LDS（一次/任务），并填充到 16 长度方便向量化
    __shared__ float s_b1[16];
    __shared__ float s_w2[16];

    int sleep_cycles = MIN_SLEEP_CYCLES;

    while (true) {
        if (threadIdx.x == 0) {
            wait_vmem_all_complete();
            // 用“获取”语义读，防止过期缓存（HIP/Clang 下 volatile 已经很好用）
            s_quit = *quit_flag;
            int has = (*task_flag == 1);
            s_has_work = has;
            s_bs = has ? *batch_size : 0;
        }
        __syncthreads();
        if (s_quit) break;

        if (s_has_work) {
            sleep_cycles = MIN_SLEEP_CYCLES;

            // 仅一次将 b1/w2 进 LDS；10..15 置 0，避免分支
            if (threadIdx.x < 16) {
                s_b1[threadIdx.x] = (threadIdx.x < 10) ? b1[threadIdx.x] : 0.0f;
                s_w2[threadIdx.x] = (threadIdx.x < 10) ? w2[threadIdx.x] : 0.0f;
            }
            __syncthreads();

            const int bs = s_bs;
            const int stride = gridDim.x * blockDim.x;

            for (int i = tid; i < bs; i += stride) {
                const float* __restrict__ x = inputs + i * NR_FEAT;

                // 10 维隐藏层累加器，放寄存器
                float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0;
                float acc5 = 0, acc6 = 0, acc7 = 0, acc8 = 0, acc9 = 0;

                // k 主循环：4 个特征一组（float4）
                int k = 0;
#pragma unroll 2
                for (; k + 3 < NR_FEAT; k += 4) {
                    // 批量取 4 个输入（自然对齐下编译器会生成 vector load）
                    float4 xv = *reinterpret_cast<const float4*>(x + k);
                    // 每个特征对应 10 个权重：w1[(k+p)*10 + j] 连续内存，cache 友好
#pragma unroll
                    for (int j = 0; j < 10; ++j) {
                        float w0 = w1[(k + 0) * 10 + j];
                        float w1v = w1[(k + 1) * 10 + j];
                        float w2v = w1[(k + 2) * 10 + j];
                        float w3v = w1[(k + 3) * 10 + j];
                        // 展开到 10 条 acc 上；fmaf 生成 FMA
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
                // 处理尾部（0~3 个特征）
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

                // 加偏置 + ReLU（用 LDS 中的 b1）
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

                // 第二层点积（w2 在 LDS）
                float out =
                    acc0 * s_w2[0] + acc1 * s_w2[1] + acc2 * s_w2[2] +
                    acc3 * s_w2[3] + acc4 * s_w2[4] + acc5 * s_w2[5] +
                    acc6 * s_w2[6] + acc7 * s_w2[7] + acc8 * s_w2[8] +
                    acc9 * s_w2[9] + b2;

                results[i] = out;
            }

            __syncthreads();
            if (tid == 0) {
                __threadfence_system();          // 先保证 results 可见
                atomicExch((int*)task_flag, 0);  // 再清旗（release）
                // 通常无需第二次 threadfence；Host 侧读旗后再读结果要用 acquire
            }
        } else {
            // 无任务：短轮询 + 指数退避
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
            // 精确睡眠：把剩余周期分段消化
            int remaining = sleep_cycles;
            while (remaining >= 1024) { __builtin_amdgcn_s_sleep(1024); remaining -= 1024; }
            if (remaining >= 512)      { __builtin_amdgcn_s_sleep(512);  remaining -= 512;  }
            if (remaining >= 256)      { __builtin_amdgcn_s_sleep(256);  remaining -= 256;  }
            if (remaining > 0)         { __builtin_amdgcn_s_sleep(64); } // 收个尾
#else
            // 其他平台：可换 __nanosleep 或空转
#endif
            long next = (long)sleep_cycles * BACKOFF_FACTOR;
            sleep_cycles = (next > MAX_SLEEP_CYCLES) ? MAX_SLEEP_CYCLES : (int)next;
        }
    }
}