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



#ifndef __MLLB_HELPERS_H
#define __MLLB_HELPERS_H

#ifdef __KERNEL__
// System includes
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>
// HIP driver
#include <hip_runtime_api_mini.h>
#include "lake_shm.h"
// CUDAdriver
#include "cuda.h"
#define LLU "%llu"
#define LLD "%lld"

#else
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#define u64 uint64_t
#define usleep_range(X,Y) sleep(X/1000000)
#define LLU "%lu"
#define LLD "%ld"
#define vmalloc(X) malloc(X)
#define vfree(X) free((void*)X)

static inline u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, 0);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
#define ktime_get_ns() get_tsns()
#define kernel_fpu_begin() (void)0
#define kernel_fpu_end() (void)0

#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <string.h>
// Include CUDA header for CUDA-specific types (if needed for CUDA platform)
// Note: On AMD/ROCm platform, these types may not be available
// Uncomment the following line if compiling for CUDA platform:
// #include "cuda.h"
#endif

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO

#ifdef __KERNEL__
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)
#else
#define PRINT(verbosity, ...) do { if (1) printf(__VA_ARGS__); } while (0)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#endif

/*static inline hipError_t check_error(hipError_t error, const char* error_str, int line)
{
	//if (error != hipSuccess) 
        if (1) 
        {
        #ifdef __KERNEL__
        if (error != hipSuccess)
        printk(KERN_ERR "ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        else
        printk(KERN_ERR "GOOD! %d (line %d): %s\n", error, line, error_str);
        #else
        printf("ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        #endif
	}       
	return error;
}*/

static inline hipError_t check_error(hipError_t error, const char* error_str, int line)
{
	if (error != hipSuccess) {
        #ifdef __KERNEL__
        printk(KERN_ERR "ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        #else
        printf("ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        #endif
	}
	return error;
}

void gpu_init(int dev, hipCtx_t *cuctx);
void gpu_get_cufunc(const char* cubin, const char* kname, hipFunction_t *func);
void gpu_setup(int n_inputs, void **d_inputs, void **d_w1, void **d_b1, void **d_w2, void **d_results);
void gpu_clean(void *d_inputs, void *d_w1, void *d_b1, void *d_w2, void *d_results);
void gpu_setup_inputs(void *d_inputs, int* inputs, int n);
//float gpu_inference();
int gpu_inference_many(hipFunction_t func, int n_inputs,
        void *d_inputs, void *d_w1, void *d_b1, void *d_w2, float b2, void *d_results, int sync);
int gpu_get_result(int n_inputs, void *d_results, float* outs);
void gpu_setup_zerocopy(int n_inputs, void **d_w1, void **d_b1, void **d_w2);

void gpu_init_cuda(int dev, CUcontext* cuctx);
void gpu_get_cufunc_cuda(char* cubin, char* kname, CUfunction *func);
int gpu_inference_many_cuda(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results, int sync);
void gpu_setup_cuda(int n_inputs, CUdeviceptr *d_inputs, CUdeviceptr *d_w1, CUdeviceptr *d_b1, CUdeviceptr *d_w2, CUdeviceptr *d_results);
void gpu_clean_cuda(CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results);
void gpu_setup_inputs_cuda(CUdeviceptr d_inputs, int* inputs, int n);
int gpu_get_result_cuda(int n_inputs, CUdeviceptr d_results, float* outs);
#endif