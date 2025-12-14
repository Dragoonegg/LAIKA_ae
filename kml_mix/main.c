/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 *
 * Original work:
 *   Copyright (C) 2022–2024 Henrique Fingler
 *   Copyright (C) 2022–2024 Isha Tarte
 *
 * Modifications and adaptations for LAIKA:
 *   Copyright (C) 2024-2025 Haoming Zhuo
 *
 * This file is adapted from the original LAKE kernel module.
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
#ifdef __KERNEL__
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include <linux/string.h>
#include "lake_shm.h"
#include "cpu.h"
#else

#define kava_free(X) free(X)
#define kava_alloc(X) malloc(X)
#define vfree(X) free(X)
#define vmalloc(X) malloc(X)
#include <stdint.h>
#include <stdio.h>
#define u64 uint64_t
#include <unistd.h>
#define usleep_range(X,Y) sleep(X/1000)
#include <sys/time.h>
#include <string.h>
u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}

#define ktime_get_ns() get_tsns()
#endif

#include "weights.h"
#include "helpers.h"
#include "cpu.h"
//#include <asm/fpu/api.h>

// Add missing function declarations
#ifndef __KERNEL__
// Function declarations for userspace mode
void setup_cpu(void);
void setup_input(int batch_size);
int cpu_predict_readahead_class(int batch_size);
void cleanup(void);

// PRINT macro definition for userspace mode
#ifndef PRINT
#define PRINT(...) printf(__VA_ARGS__)
#endif
#endif

static char *cubin_path = "kml.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to kml.cubin, default ./kml.cubin");
#endif

static char *hsaco_path = "kml.hsaco";
#ifdef __KERNEL__
module_param(hsaco_path, charp, 0444);
MODULE_PARM_DESC(hsaco_path, "The path to kml.hsaco, default ./kml.hsaco");
#endif

#define RUNS 1000
#define CPURUNS 10
#define USE_CUDA_SYNC 1


static hipFunction_t normalize_fused, forward_fused, fully_fused_forward;
static hipFunction_t normalize_fused_persistent, fused_forward_persistent, fully_fused_persistent;
static hipFunction_t fully_fused_persistent_optimized;


static int run_cpu(void) {
    return 0;
}

static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_cols;

void *d_w0, *d_b0, *d_w1, *d_b1, *d_w2, *d_b2, *d_out0, *d_out1, *d_out2, *d_input, *d_result_cols, *d_intital_stats;
static float* batch_input;
static int *result;

//refactor here
int readahead_online_data_cols = 5;
int readahead_online_data_rows = 1;
void *d_readahead_norm_online_data;
//void* cu_get_average, cu_get_variance, cu_matrix_map, cu_normalize_data, cu_matrix_transpose,
//    cu_matrix_mult, cu_add_bias, cu_matrix_argmax;
static hipFunction_t normalize_fused, forward_fused;

//readahead_normalized_online_data
void *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;
void *bias;
void *wt0, *wt1, *wt2;

static void setup_gpu(int batch_size) {
    float *w0, *w1, *w2, *b0, *b1, *b2;
    w0_rows = 15;
    w0_cols = 5;
    b0_rows = 15;
    b0_cols = 1;
    w1_rows = 5;
    w1_cols = 15;
    b1_rows = 5;
    b1_cols = 1; 
    w2_rows = 4;
    w2_cols = 5;
    b2_rows = 4;
    b2_cols = 1;

    

    w0 = &w0_arr[0][0];
    float *kbfuf_w0 = (float*) kava_alloc(w0_rows * w0_cols * sizeof(float));
    memcpy(kbfuf_w0, w0, w0_rows * w0_cols * sizeof(float));

    b0 = &b0_arr[0][0];
    float *kbfuf_b0 = (float*) kava_alloc(b0_rows * b0_cols * sizeof(float));
    memcpy(kbfuf_b0, b0, b0_rows * b0_cols * sizeof(float));

    w1 = &w1_arr[0][0];
    float *kbfuf_w1 = (float*) kava_alloc(w1_rows * w1_cols * sizeof(float));
    memcpy(kbfuf_w1, w1, w1_rows * w1_cols * sizeof(float));

    b1 = &b1_arr[0][0];
    float *kbfuf_b1 = (float*) kava_alloc(b1_rows * b1_cols * sizeof(float));
    memcpy(kbfuf_b1, b1, b1_rows * b1_cols * sizeof(float));

    w2 = &w2_arr[0][0];
    float *kbfuf_w2 = (float*) kava_alloc(w2_rows * w2_cols * sizeof(float));
    memcpy(kbfuf_w2, w2, w2_rows * w2_cols * sizeof(float));

    b2 = &b2_arr[0][0];
    float *kbfuf_b2 = (float*) kava_alloc(b2_rows * b2_cols * sizeof(float));
    memcpy(kbfuf_b2, b2, b2_rows * b2_cols * sizeof(float));

    float *stats = &intial_stats[0];
    float *kbfuf_stats = (float*) kava_alloc(input_cols * sizeof(float));
    memcpy(kbfuf_stats, stats, input_cols * sizeof(float));

    int input_features = 5;
    input_cols = input_features;
    check_error(hipMalloc((void**)&d_w0, sizeof(float) *w0_rows * w0_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_w0, kbfuf_w0, sizeof(float) * w0_rows * w0_cols), "hipMemcpyHtoD", __LINE__);

    check_error(hipMalloc((void**)&d_b0, sizeof(float) *b0_rows * b0_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_b0, kbfuf_b0, sizeof(float) * b0_rows * b0_cols), "hipMemcpyHtoD", __LINE__);

    check_error(hipMalloc((void**)&d_w1, sizeof(float) *w1_rows * w1_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_w1, kbfuf_w1, sizeof(float) * w1_rows * w1_cols), "hipMemcpyHtoD", __LINE__);

    check_error(hipMalloc((void**)&d_b1, sizeof(float) *b1_rows * b1_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_b1, kbfuf_b1, sizeof(float) * b1_rows * b1_cols), "hipMemcpyHtoD", __LINE__);

    check_error(hipMalloc((void**)&d_w2, sizeof(float) *w2_rows * w2_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_w2, kbfuf_w2, sizeof(float) * w2_rows * w2_cols), "hipMemcpyHtoD", __LINE__);
    
    check_error(hipMalloc((void**)&d_b2, sizeof(float) *b2_rows * b2_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_b2, kbfuf_b2, sizeof(float) * b2_rows * b2_cols), "hipMemcpyHtoD", __LINE__);

    /*float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    //float batch_input[batch_size][5];
    int i ,j;
    for(i = 0; i < batch_size; i++) {
        for(j = 0 ; j < 5; j++) {
            batch_input[i*5 + j] = input[j];
        }
    }*/

 

    check_error(hipMalloc((void**)&d_intital_stats, sizeof(float) *input_cols), "hipMalloc ", __LINE__);
    check_error(hipMemcpyHtoD(d_intital_stats, kbfuf_stats, sizeof(float) * input_cols), "hipMemcpyHtoD", __LINE__);
    
    // refactor below
    

    // readahead_normalized_online_data
    check_error(hipMalloc((void**)&local_average, 
        sizeof(float)  * readahead_online_data_cols), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**)&local_std_dev, 
        sizeof(float)  * readahead_online_data_cols), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**)&local_variance, 
        sizeof(float) * readahead_online_data_cols), "hipMalloc ", __LINE__);

    

    // linear_layer_forward
    check_error(hipMalloc((void**)&wt0, 
        sizeof(float) * w0_cols * w0_rows), "hipMalloc ", __LINE__);
    // check_error(hipMalloc((void**)&wx0, 
    //     sizeof(float) * batch_size * w0_rows), "hipMalloc ", __LINE__);

    check_error(hipMalloc((void**)&wt1, 
        sizeof(float) * w1_cols * w1_rows), "hipMalloc ", __LINE__);
    // check_error(hipMalloc((void**)&wx1, 
    //     sizeof(float) * batch_size * w1_rows), "hipMalloc ", __LINE__);

    check_error(hipMalloc((void**)&wt2, 
        sizeof(float) * w2_cols * w2_rows), "hipMalloc ", __LINE__);
    // check_error(hipMalloc((void**)&wx2, 
    //     sizeof(float) * batch_size * w2_rows), "hipMalloc ", __LINE__);

    kava_free(kbfuf_w0);
    kava_free(kbfuf_w1);
    kava_free(kbfuf_w2);
    kava_free(kbfuf_b0);
    kava_free(kbfuf_b1);
    kava_free(kbfuf_b2);
    kava_free(kbfuf_stats);
}

static void setup_gpu_cuda(int batch_size) {
    float *w0, *w1, *w2, *b0, *b1, *b2;
    w0_rows = 15;
    w0_cols = 5;
    b0_rows = 15;
    b0_cols = 1;
    w1_rows = 5;
    w1_cols = 15;
    b1_rows = 5;
    b1_cols = 1; 
    w2_rows = 4;
    w2_cols = 5;
    b2_rows = 4;
    b2_cols = 1;

    

    w0 = &w0_arr[0][0];
    float *kbfuf_w0 = (float*) kava_alloc(w0_rows * w0_cols * sizeof(float));
    memcpy(kbfuf_w0, w0, w0_rows * w0_cols * sizeof(float));

    b0 = &b0_arr[0][0];
    float *kbfuf_b0 = (float*) kava_alloc(b0_rows * b0_cols * sizeof(float));
    memcpy(kbfuf_b0, b0, b0_rows * b0_cols * sizeof(float));

    w1 = &w1_arr[0][0];
    float *kbfuf_w1 = (float*) kava_alloc(w1_rows * w1_cols * sizeof(float));
    memcpy(kbfuf_w1, w1, w1_rows * w1_cols * sizeof(float));

    b1 = &b1_arr[0][0];
    float *kbfuf_b1 = (float*) kava_alloc(b1_rows * b1_cols * sizeof(float));
    memcpy(kbfuf_b1, b1, b1_rows * b1_cols * sizeof(float));

    w2 = &w2_arr[0][0];
    float *kbfuf_w2 = (float*) kava_alloc(w2_rows * w2_cols * sizeof(float));
    memcpy(kbfuf_w2, w2, w2_rows * w2_cols * sizeof(float));

    b2 = &b2_arr[0][0];
    float *kbfuf_b2 = (float*) kava_alloc(b2_rows * b2_cols * sizeof(float));
    memcpy(kbfuf_b2, b2, b2_rows * b2_cols * sizeof(float));

    float *stats = &intial_stats[0];
    float *kbfuf_stats = (float*) kava_alloc(input_cols * sizeof(float));
    memcpy(kbfuf_stats, stats, input_cols * sizeof(float));

    int input_features = 5;
    input_cols = input_features;
    check_error(cuMemAlloc((CUdeviceptr*) &d_w0, sizeof(float) *w0_rows * w0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w0, kbfuf_w0, sizeof(float) * w0_rows * w0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b0, sizeof(float) *b0_rows * b0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b0, kbfuf_b0, sizeof(float) * b0_rows * b0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w1, sizeof(float) *w1_rows * w1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w1, kbfuf_w1, sizeof(float) * w1_rows * w1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b1, sizeof(float) *b1_rows * b1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b1, kbfuf_b1, sizeof(float) * b1_rows * b1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w2, sizeof(float) *w2_rows * w2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w2, kbfuf_w2, sizeof(float) * w2_rows * w2_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b2, sizeof(float) *b2_rows * b2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b2, kbfuf_b2, sizeof(float) * b2_rows * b2_cols), "cuMemcpyHtoD", __LINE__);
    
    float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    //float batch_input[batch_size][5];
    int i ,j;
    for(i = 0; i < batch_size; i++) {
        for(j = 0 ; j < 5; j++) {
            batch_input[i*5 + j] = input[j];
        }
    }

    check_error(cuMemAlloc((CUdeviceptr*) &d_input, sizeof(float) *input_features * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_result_cols, sizeof(int) * batch_size), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_intital_stats, sizeof(float) *input_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_intital_stats, kbfuf_stats, sizeof(float) * input_cols), "cuMemcpyHtoD", __LINE__);
    
    // refactor below
    check_error(cuMemAlloc((CUdeviceptr*) &d_readahead_norm_online_data, 
        sizeof(float) *readahead_online_data_cols * batch_size), "cuMemAlloc ", __LINE__);

    // readahead_normalized_online_data
    check_error(cuMemAlloc((CUdeviceptr*) &local_average, 
        sizeof(float)  * readahead_online_data_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &local_std_dev, 
        sizeof(float)  * readahead_online_data_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &local_variance, 
        sizeof(float) * readahead_online_data_cols), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &readahead_norm_online_data_last_values, 
        sizeof(float) * readahead_online_data_cols * batch_size), "cuMemAlloc ", __LINE__);

    // autodiff_forward
    check_error(cuMemAlloc((CUdeviceptr*) &d_out0, sizeof(float) * w0_rows * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_out1, sizeof(float) * w1_rows * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_out2,         sizeof(float) * w2_rows * batch_size), "cuMemAlloc ", __LINE__);

    // linear_layer_forward
    check_error(cuMemAlloc((CUdeviceptr*) &wt0, 
        sizeof(float) * w0_cols * w0_rows), "cuMemAlloc ", __LINE__);
    // check_error(cuMemAlloc((CUdeviceptr*) &wx0, 
    //     sizeof(float) * batch_size * w0_rows), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &wt1, 
        sizeof(float) * w1_cols * w1_rows), "cuMemAlloc ", __LINE__);
    // check_error(cuMemAlloc((CUdeviceptr*) &wx1, 
    //     sizeof(float) * batch_size * w1_rows), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &wt2, 
        sizeof(float) * w2_cols * w2_rows), "cuMemAlloc ", __LINE__);
    // check_error(cuMemAlloc((CUdeviceptr*) &wx2, 
    //     sizeof(float) * batch_size * w2_rows), "cuMemAlloc ", __LINE__);

    kava_free(kbfuf_w0);
    kava_free(kbfuf_w1);
    kava_free(kbfuf_w2);
    kava_free(kbfuf_b0);
    kava_free(kbfuf_b1);
    kava_free(kbfuf_b2);
    kava_free(kbfuf_stats);
}


static void copy_batch_inputs(int batch_size) {
    float *kbfuf_input = (float*) kava_alloc(sizeof(float) * input_cols * batch_size);
    memcpy(kbfuf_input, batch_input, sizeof(float) * input_cols * batch_size);
    check_error(hipMemcpyHtoD(d_input, kbfuf_input, sizeof(float) * input_cols * batch_size), "hipMemcpyHtoD", __LINE__);
}

static void get_result_batch(int batch_size) {
    check_error(hipMemcpyDtoH(result, d_result_cols, sizeof(int) * batch_size), "hipMemcpyDtoH", __LINE__);
}



void clean_batch(void) {
    hipFree(d_w0);
    hipFree(d_w1);
    hipFree(d_w2);
    hipFree(d_b0);
    hipFree(d_b1);
    hipFree(d_b2);
    hipFree(d_out0);
    hipFree(d_out1);
    hipFree(d_out2);
	hipFree(d_input);

    kava_free(batch_input);
    kava_free(result);

    //refactor below
    hipFree(local_average);
    hipFree(local_std_dev);
    hipFree(local_variance);
    hipFree(readahead_norm_online_data_last_values);

    hipFree(wt0);
    //hipFree(wx0);
    hipFree(wt1);
    //hipFree(wx1);
    hipFree(wt2);
    //hipFree(wx2);
}



void autodiff_forward(int batch_size, int sync) { 


     void *args[] = {
		&d_readahead_norm_online_data, &d_result_cols, &batch_size,
        &d_w0, &d_b0, &wt0,
        &d_w1, &d_b1, &wt1, 
        &d_w2, &d_b2, &wt2,
        &d_out0, &d_out1, &d_out2
	};
    
    int zg = sync == 0 ? 1 : 69; 
    check_error(hipModuleLaunchKernel(forward_fused, 
				batch_size, 1, zg,          //blocks
				16, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"hipModuleLaunchKernel", __LINE__);
    if (USE_CUDA_SYNC == 1) {
        check_error(hipDeviceSynchronize(), "hipDeviceSynchronize", __LINE__);
    }
}

void readahead_normalized_online_data(int batch_size) {
   

    // TODO: uncomment above and find why values are different
    
    void *fargs[] = {
        &batch_size, &d_input, &d_intital_stats, &local_average,
        &readahead_norm_online_data_last_values, &local_variance, &d_readahead_norm_online_data
	};

    int blocks = (batch_size+159) / 160; //ceil
    int tpb = batch_size < 160 ? 32 : 160; //at least 32 threads, at most 160 (32*5)

    check_error(hipModuleLaunchKernel(normalize_fused, 
				blocks, 1, 1,          //blocks
				tpb, 1, 1,   //threads per block
				0,   //shared mem
                NULL, fargs, NULL),
			"hipModuleLaunchKernel", __LINE__);
    
}   

void predict_readahead_class(int batch_size, int sync) {
    readahead_normalized_online_data(batch_size);
    autodiff_forward(batch_size, sync);
}

void predict_readahead_class_cuda(int batch_size, int sync) {
    readahead_normalized_online_data_cuda(batch_size);
    autodiff_forward_cuda(batch_size, sync);
}

void readahead_normalized_online_data_cuda(int batch_size) {
    // TODO: uncomment above and find why values are different
    void *fargs[] = {
        &batch_size, &d_input, &d_intital_stats, &local_average,
        &readahead_norm_online_data_last_values, &local_variance, &d_readahead_norm_online_data
	};

    int blocks = (batch_size+159) / 160; //ceil
    int tpb = batch_size < 160 ? 32 : 160; //at least 32 threads, at most 160 (32*5)

    check_error(cuLaunchKernel(normalize_fused, 
				blocks, 1, 1,          //blocks
				tpb, 1, 1,   //threads per block
				0,   //shared mem
                NULL, fargs, NULL),
			"cuLaunchKernel", __LINE__);
    
}   

void autodiff_forward_cuda(int batch_size, int sync) { 

     void *args[] = {
		&d_readahead_norm_online_data, &d_result_cols, &batch_size,
        &d_w0, &d_b0, &wt0,
        &d_w1, &d_b1, &wt1, 
        &d_w2, &d_b2, &wt2,
        &d_out0, &d_out1, &d_out2
	};
    int zg = sync == 0 ? 1 : 69; 
    check_error(cuLaunchKernel(forward_fused, 
				batch_size, 1, zg,          //blocks
				16, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    if (USE_CUDA_SYNC == 1) {
        check_error(cuCtxSynchronize(), "cudaDeviceSynchronize", __LINE__);
    }
}

static void copy_batch_inputs_cuda(int batch_size) {
    float *kbfuf_input = (float*) kava_alloc(sizeof(float) * input_cols * batch_size);
    memcpy(kbfuf_input, batch_input, sizeof(float) * input_cols * batch_size);
    check_error(cuMemcpyHtoD(d_input, kbfuf_input, sizeof(float) * input_cols * batch_size), "cuMemcpyHtoD", __LINE__);
    kava_free(kbfuf_input);
}

static void get_result_batch_cuda(int batch_size) {
    check_error(cuMemcpyDtoH(result, d_result_cols, sizeof(int) * batch_size), "cuMemcpyDtoH", __LINE__);
}

void clean_batch_cuda(void) {
    cuMemFree(d_w0);
    cuMemFree(d_w1);
    cuMemFree(d_w2);
    cuMemFree(d_b0);
    cuMemFree(d_b1);
    cuMemFree(d_b2);
    cuMemFree(d_out0);
    cuMemFree(d_out1);
    cuMemFree(d_out2);
	cuMemFree(d_input);
    cuMemFree(d_result_cols);
    cuMemFree(d_intital_stats);
    cuMemFree(d_readahead_norm_online_data);

    kava_free(batch_input);
    kava_free(result);

    //refactor below
    cuMemFree(local_average);
    cuMemFree(local_std_dev);
    cuMemFree(local_variance);
    cuMemFree(readahead_norm_online_data_last_values);

    cuMemFree(wt0);
    //cuMemFree(wx0);
    cuMemFree(wt1);
    //cuMemFree(wx1);
    cuMemFree(wt2);
    //cuMemFree(wx2);
}

static int run_dgpu(void) {
    int i, j, x;
    const int n = 2048;
    int batch_sizes[] = {16,1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,2048,4096};
    int n_batches = sizeof(batch_sizes)/sizeof(int);

    int batch_size;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
  
    CUcontext cuContext;
    gpu_init_cuda(0, &cuContext);
    gpu_get_cufunc_cuda(cubin_path, "_Z15normalize_fusediPfS_S_S_S_S_", &normalize_fused);
    gpu_get_cufunc_cuda(cubin_path, "_Z13fused_forwardPfPiiS_S_S_S_S_S_S_S_S_S_S_S_", &forward_fused);
    
    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

        batch_input = (float*) kava_alloc(batch_size * 5 * sizeof(float));
        result = (int*) kava_alloc(batch_size * sizeof(int));
        setup_gpu_cuda(batch_size);
        //copy_batch_inputs_cuda(batch_size);
        //predict_readahead_class_cuda(batch_size, 0);
       // cuCtxSynchronize();
       // predict_readahead_class_cuda(batch_size, 0);
        cuCtxSynchronize();

        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            copy_batch_inputs_cuda(batch_size);
            predict_readahead_class_cuda(batch_size, 0);
            get_result_batch_cuda(batch_size);
            t_stop = ktime_get_ns();

            total_run_times[j] = (t_stop - t_start);
	    }

	    avg = 0; avg_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);

#ifdef __KERNEL__
        PRINT("KML_dGPU_batch_%d,%lld\n", batch_size,avg_total);
#else
        printf("GPU batch_%d, %ld\n", batch_size, avg_total);
#endif
        clean_batch_cuda();
	}

    vfree(comp_run_times);
    vfree(total_run_times);
    
    return 0;
}



static int run_persistent(void) {
    int i, j, x;
    //const int n = 1024;
    
    int batch_sizes[] = {16,1,2,4,8,16,32,64,128,256};
   // int batch_sizes[] = {1,8};
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch = batch_sizes[n_batches-1];

    int batch_size;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    
  
    hipCtx_t cuContext;
    gpu_init(0, &cuContext);
    //gpu_get_cufunc(hsaco_path, "_Z15normalize_fusediPfS_S_S_S_S_", &normalize_fused);
    //gpu_get_cufunc(hsaco_path, "_Z13fused_forwardPfPiiS_S_S_S_S_S_S_S_S_S_S_S_", &forward_fused);
    gpu_get_cufunc(hsaco_path, "_Z26normalize_fused_persistentiPfS_S_S_S_S_PiS0_", &normalize_fused_persistent);
    gpu_get_cufunc(hsaco_path, "_Z24fused_forward_persistentPfPiiS_S_S_S_S_S_S_S_S_S_S_S_S0_S0_", &fused_forward_persistent);
    setup_gpu(0);
    // Create two different streams
    hipStream_t stream1, stream2;
    #ifdef __KERNEL__
    check_error(hipStreamCreate(&stream1,0), "hipStreamCreate stream1", __LINE__);
    check_error(hipStreamCreate(&stream2,0), "hipStreamCreate stream2", __LINE__);
    #else
    check_error(hipStreamCreate(&stream1), "hipStreamCreate stream1", __LINE__);
    check_error(hipStreamCreate(&stream2), "hipStreamCreate stream2", __LINE__);
    #endif
    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    // persistent kernel synchronization flag
    int* h_task_flag_1 = (int*)kava_alloc(sizeof(int));
    int* h_quit_flag = (int*)kava_alloc(sizeof(int));
    hipHostRegister(h_task_flag_1, sizeof(int), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
    hipHostRegister(h_quit_flag, sizeof(int), hipHostRegisterMapped);
    void* d_task_flag_1; hipHostGetDevicePointer(&d_task_flag_1, h_task_flag_1, 0);
    void* d_quit_flag; hipHostGetDevicePointer(&d_quit_flag, h_quit_flag, 0);


    // Track resources allocated within the loop
    void *h_inputs_mapped = NULL;
    
    for (i = 0 ; i < n_batches ; i++) {
        // Reset synchronization flag before each batch starts
        *h_task_flag_1 = 0; *h_quit_flag = 0;
        batch_size = batch_sizes[i];
        
        // // Free resources allocated in the previous iteration (if any)
        // if (d_readahead_norm_online_data) {
        //     hipFree(d_readahead_norm_online_data);
        //     d_readahead_norm_online_data = NULL;
        // }
        // if (d_out0) {
        //     hipFree(d_out0);
        //     d_out0 = NULL;
        // }
        // if (d_out1) {
        //     hipFree(d_out1);
        //     d_out1 = NULL;
        // }
        // if (d_out2) {
        //     hipFree(d_out2);
        //     d_out2 = NULL;
        // }
        // if (h_inputs_mapped) {
        //     hipHostUnregister(h_inputs_mapped);
        //     kava_free(h_inputs_mapped);
        //     h_inputs_mapped = NULL;
        // }
        // if (result) {
        //     hipHostUnregister(result);
        //     kava_free(result);
        //     result = NULL;
        // }
        
        // Initialize space for intermediate function values
        check_error(hipMalloc((void**)&d_readahead_norm_online_data, sizeof(float) *readahead_online_data_cols * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_out0, sizeof(float) * w0_rows * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_out1, sizeof(float) * w1_rows * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_out2, sizeof(float) * w2_rows * batch_size), "hipMalloc ", __LINE__);
        // Map input memory
        void *d_inputs_mapped;
        h_inputs_mapped = kava_alloc(batch_size * 5 * sizeof(float));
        hipHostRegister(h_inputs_mapped, batch_size * 5 * sizeof(float), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
        hipHostGetDevicePointer(&d_inputs_mapped, h_inputs_mapped, 0);
        float *linear_inputs = (float*)h_inputs_mapped;

        // Map output memory
        void *d_result_mapped;
        result = (int*) kava_alloc(batch_size * sizeof(int));
        hipHostRegister(result, batch_size * sizeof(int), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
        hipHostGetDevicePointer(&d_result_mapped, result, 0);
        hipDeviceSynchronize();
        // Generate some data and place it in h_inputs_mapped
        float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};   
        for(int k = 0; k< batch_size; k++) {
            for(j = 0 ; j < 5; j++) {
                linear_inputs[k*5 + j] = input[j];
            }
        }
        #ifdef __KERNEL__
        mb();
        #else
        __sync_synchronize();
        #endif
        // Launch persistent kernel 1 - run on stream1
         void *fargs[] = {
            &batch_size, &d_inputs_mapped, &d_intital_stats, &local_average,&readahead_norm_online_data_last_values, &local_variance, &d_readahead_norm_online_data,&d_task_flag_1, &d_quit_flag};
            int blocks = (batch_size+159) / 160; //ceil
            int tpb = batch_size < 160 ? 32 : 160; //at least 32 threads, at most 160 (32*5)
            check_error(hipModuleLaunchKernel(normalize_fused_persistent, 
				blocks, 1, 1,          //blocks
				tpb, 1, 1,   //threads per block
				0,   //shared mem
                stream1, fargs, NULL),
			"hipModuleLaunchKernel stream1", __LINE__);
        // Launch persistent kernel 2 - run on stream2
        void *args[] = {
		    &d_readahead_norm_online_data, &d_result_mapped, &batch_size,
            &d_w0, &d_b0, &wt0,
            &d_w1, &d_b1, &wt1, 
            &d_w2, &d_b2, &wt2,
            &d_out0, &d_out1, &d_out2, &d_task_flag_1, &d_quit_flag
	        };
            int sync=0;
            int zg = sync == 0 ? 1 : 69; 
            check_error(hipModuleLaunchKernel( fused_forward_persistent, 
				batch_size, 1, zg,          //blocks
				16, 1, 1,   //threads per block
				0,   //shared mem
                stream2, args, NULL),
			"hipModuleLaunchKernel stream2", __LINE__);
                // do some warmup
            for (j = 0 ; j < 100 ; j++) {
                 *h_task_flag_1 = 1;
                 while (*h_task_flag_1 != 3) {
                    __sync_synchronize();
                }
            }


        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            // -------------------------------RUN-1-------------------------------
             *h_task_flag_1 = 1;
             while (*h_task_flag_1 != 3) {
            __sync_synchronize();   // PRINT("h_task_flag_1 = %d,h_quit_flag = %d\n", *h_task_flag_1, *h_quit_flag);
            }
            // -------------------------------RUN-2-------------------------------
             t_stop = ktime_get_ns();
            total_run_times[j] = (t_stop - t_start);
	    }

        *h_quit_flag = 1;
        hipStreamSynchronize(stream1);
        hipStreamSynchronize(stream2);
        
	    avg = 0; avg_total = 0;
        for (j = 0 ; j < RUNS; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
    

#ifdef __KERNEL__
        PRINT("KML_APU_PK_batch_%d,%lu\n", batch_size, avg_total);
#else
        printf("KML_APU_PK_batch_%d, %lu\n", batch_size, avg_total);
#endif
	}
    
    // Free resources allocated in the last iteration
    if (d_readahead_norm_online_data) {
        hipFree(d_readahead_norm_online_data);
        d_readahead_norm_online_data = NULL;
    }
    if (d_out0) {
        hipFree(d_out0);
        d_out0 = NULL;
    }
    if (d_out1) {
        hipFree(d_out1);
        d_out1 = NULL;
    }
    if (d_out2) {
        hipFree(d_out2);
        d_out2 = NULL;
    }
    if (h_inputs_mapped) {
        hipHostUnregister(h_inputs_mapped);
        kava_free(h_inputs_mapped);
        h_inputs_mapped = NULL;
    }
    if (result) {
        hipHostUnregister(result);
        kava_free(result);
        result = NULL;
    }

    // Free resources allocated outside the loop
    if (h_task_flag_1) {
        hipHostUnregister(h_task_flag_1);
        kava_free(h_task_flag_1);
        h_task_flag_1 = NULL;
    }
    if (h_quit_flag) {
        hipHostUnregister(h_quit_flag);
        kava_free(h_quit_flag);
        h_quit_flag = NULL;
    }
    if (comp_run_times) {
        vfree(comp_run_times);
        comp_run_times = NULL;
    }
    if (total_run_times) {
        vfree(total_run_times);
        total_run_times = NULL;
    }
    
    check_error(hipStreamDestroy(stream1), "hipStreamDestroy stream1", __LINE__);
    check_error(hipStreamDestroy(stream2), "hipStreamDestroy stream2", __LINE__);
    
    return 0;
}



static int run_apu(void) {
    int i, j, x;
    //const int n = 1024;
    //int batch_sizes[] = {1,2,4,8};
    int batch_sizes[] = {16,1,2,4,8,16,32,64,128,256,512,1024,2048,4096};
    //int batch_sizes[] = {8};
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch = batch_sizes[n_batches-1];

    int batch_size;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
  
    hipCtx_t cuContext;
    gpu_init(0, &cuContext);

    gpu_get_cufunc(hsaco_path, "_Z15normalize_fusediPfS_S_S_S_S_", &normalize_fused);
    gpu_get_cufunc(hsaco_path, "_Z13fused_forwardPfPiiS_S_S_S_S_S_S_S_S_S_S_S_", &forward_fused);
    setup_gpu(0);
    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        
        // // Free resources allocated in the previous iteration (if any)
        // if (d_input) {
        //     hipFree(d_input);
        //     d_input = NULL;
        // }
        // if (d_result_cols) {
        //     hipFree(d_result_cols);
        //     d_result_cols = NULL;
        // }
        // if (d_readahead_norm_online_data) {
        //     hipFree(d_readahead_norm_online_data);
        //     d_readahead_norm_online_data = NULL;
        // }
        // if (readahead_norm_online_data_last_values) {
        //     hipFree(readahead_norm_online_data_last_values);
        //     readahead_norm_online_data_last_values = NULL;
        // }
        // if (d_out0) {
        //     hipFree(d_out0);
        //     d_out0 = NULL;
        // }
        // if (d_out1) {
        //     hipFree(d_out1);
        //     d_out1 = NULL;
        // }
        // if (d_out2) {
        //     hipFree(d_out2);
        //     d_out2 = NULL;
        // }
        // if (result) {
        //     hipHostUnregister(result);
        //     kava_free(result);
        //     result = NULL;
        // }
        
        check_error(hipMalloc((void**)&d_input, sizeof(float) *5 * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_result_cols, sizeof(int) * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_readahead_norm_online_data, 
        sizeof(float) *readahead_online_data_cols * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&readahead_norm_online_data_last_values, 
        sizeof(float) * readahead_online_data_cols * batch_size), "hipMalloc ", __LINE__);
        //autodiff_forward
        check_error(hipMalloc((void**)&d_out0, sizeof(float) * w0_rows * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_out1, sizeof(float) * w1_rows * batch_size), "hipMalloc ", __LINE__);
        check_error(hipMalloc((void**)&d_out2, sizeof(float) * w2_rows * batch_size), "hipMalloc ", __LINE__);
        // Map input memory
        void *d_inputs_mapped;
        void *h_inputs_mapped;
        

        h_inputs_mapped = kava_alloc(batch_size * 5 * sizeof(float));
        
        // Map input memory
        hipError_t reg_error = hipHostRegister(h_inputs_mapped, batch_size * 5 * sizeof(float), hipHostRegisterMapped);
        if (reg_error != hipSuccess) {
            PRINT("ERROR: hipHostRegister failed with error %d\n", reg_error);
            return -1;
        }
        hipError_t get_ptr_error = hipHostGetDevicePointer(&d_inputs_mapped, h_inputs_mapped, 0);
        if (get_ptr_error != hipSuccess) {
            PRINT("ERROR: hipHostGetDevicePointer failed with error %d\n", get_ptr_error);
            return -1;
        }
        float *linear_inputs = (float*)h_inputs_mapped;
        // Generate some data and place it in h_inputs_mapped
        float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};   
        for(int k = 0; k< batch_size; k++) {
            for(j = 0 ; j < 5; j++) {
                linear_inputs[k*5 + j] = input[j];
            }
        }
        // Map output memory
        void *d_result_mapped;
        result = (int*) kava_alloc(batch_size * sizeof(int));
        hipHostRegister(result, batch_size * sizeof(int), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
        hipHostGetDevicePointer(&d_result_mapped, result, 0);
        hipDeviceSynchronize();

        for (j = 0 ; j < RUNS ; j++) {
            //get data ready
            t_start = ktime_get_ns();
            //-------------------------------RUN-1-------------------------------
            void *fargs[] = {
            &batch_size, &d_inputs_mapped, &d_intital_stats, &local_average,&readahead_norm_online_data_last_values, &local_variance, &d_readahead_norm_online_data};
            int blocks = (batch_size+159) / 160; //ceil
            int tpb = batch_size < 160 ? 32 : 160; //at least 32 threads, at most 160 (32*5)
            check_error(hipModuleLaunchKernel(normalize_fused, 
				blocks, 1, 1,          //blocks
				tpb, 1, 1,   //threads per block
				0,   //shared mem
                NULL, fargs, NULL),
			"hipModuleLaunchKernel", __LINE__);
            //-------------------------------RUN-2-------------------------------
            void *args[] = {
		    &d_readahead_norm_online_data, &d_result_mapped, &batch_size,
            &d_w0, &d_b0, &wt0,
            &d_w1, &d_b1, &wt1, 
            &d_w2, &d_b2, &wt2,
            &d_out0, &d_out1, &d_out2
	         };
             int sync=0;
            int zg = sync == 0 ? 1 : 69; 
            check_error(hipModuleLaunchKernel(forward_fused, 
				batch_size, 1, zg,          //blocks
				16, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"hipModuleLaunchKernel", __LINE__);
            if (USE_CUDA_SYNC == 1) {
            check_error(hipDeviceSynchronize(), "hipDeviceSynchronize", __LINE__);
             }
             t_stop = ktime_get_ns();
          
    
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
	    }

	    avg = 0; avg_total = 0;
        for (j = 0 ; j < RUNS; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);

#ifdef __KERNEL__
        PRINT("KML_APU_PL_batch_%d,%lu\n", batch_size, avg_total);
#else
        printf("KML_APU_PL_batch_%d,%lu\n", batch_size, avg_total);
#endif
        
        // Free resources allocated in this loop iteration
        // First unregister host memory
        if (h_inputs_mapped) {
            hipHostUnregister(h_inputs_mapped);
            kava_free(h_inputs_mapped);
            h_inputs_mapped = NULL;
        }
        if (result) {
            hipHostUnregister(result);
            kava_free(result);
            result = NULL;
        }
        // Free GPU memory
        if (d_input) {
            hipFree(d_input);
            d_input = NULL;
        }
        if (d_result_cols) {
            hipFree(d_result_cols);
            d_result_cols = NULL;
        }
        if (d_readahead_norm_online_data) {
            hipFree(d_readahead_norm_online_data);
            d_readahead_norm_online_data = NULL;
        }
        if (readahead_norm_online_data_last_values) {
            hipFree(readahead_norm_online_data_last_values);
            readahead_norm_online_data_last_values = NULL;
        }
        if (d_out0) {
            hipFree(d_out0);
            d_out0 = NULL;
        }
        if (d_out1) {
            hipFree(d_out1);
            d_out1 = NULL;
        }
        if (d_out2) {
            hipFree(d_out2);
            d_out2 = NULL;
        }
	}

    
    
    setup_cpu();
    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        setup_input(batch_size);
        for (j = 0 ; j < CPURUNS ; j++) {
            t_start = ktime_get_ns();
            x = cpu_predict_readahead_class(batch_size);
            (void)x;
            t_stop = ktime_get_ns();
            usleep_range(1000, 2000);
            total_run_times[j] = (t_stop - t_start);
	    }

	    avg_total = 0;
        for (j = 0 ; j < CPURUNS ; j++) {
            avg_total += total_run_times[j];
        }
        avg_total = avg_total / (1000*CPURUNS);

        PRINT("KML_CPU_batch_%d,%lu\n", batch_size, avg_total);
	}

    
    vfree(comp_run_times);
    vfree(total_run_times);
    
    // Free resources allocated in setup_gpu
    if (d_w0) {
        hipFree(d_w0);
        d_w0 = NULL;
    }
    if (d_w1) {
        hipFree(d_w1);
        d_w1 = NULL;
    }
    if (d_w2) {
        hipFree(d_w2);
        d_w2 = NULL;
    }
    if (d_b0) {
        hipFree(d_b0);
        d_b0 = NULL;
    }
    if (d_b1) {
        hipFree(d_b1);
        d_b1 = NULL;
    }
    if (d_b2) {
        hipFree(d_b2);
        d_b2 = NULL;
    }
    if (d_intital_stats) {
        hipFree(d_intital_stats);
        d_intital_stats = NULL;
    }
    if (local_average) {
        hipFree(local_average);
        local_average = NULL;
    }
    if (local_std_dev) {
        hipFree(local_std_dev);
        local_std_dev = NULL;
    }
    if (local_variance) {
        hipFree(local_variance);
        local_variance = NULL;
    }
    if (wt0) {
        hipFree(wt0);
        wt0 = NULL;
    }
    if (wt1) {
        hipFree(wt1);
        wt1 = NULL;
    }
    if (wt2) {
        hipFree(wt2);
        wt2 = NULL;
    }
    
    // Free GPU context
    if (cuContext) {
        hipCtxDestroy(cuContext);
        cuContext = NULL;
    }
    
    return 0;
}


#ifdef __KERNEL__

/**
 * Program main
 */
static int __init kml_init(void)
{   
   run_dgpu();
    run_apu();
    run_persistent();
	return 0;
}

static void __exit kml_fini(void)
{

}

module_init(kml_init);
module_exit(kml_fini);

MODULE_AUTHOR("Isha Tarte");
MODULE_AUTHOR("Haoming Zhuo");
MODULE_DESCRIPTION( "Adapted kernel module based on the original MLLB implementation by Isha Tarte");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0.0");
#else

int main() {
    return 0;
}

#endif
