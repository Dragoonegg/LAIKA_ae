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
#include "cuda.h"
#include "lake_shm.h"
//uspace
#else
#define kava_free(X) free(X)
#define kava_alloc(X) malloc(X)
#define vfree(X) free(X)
#define vmalloc(X) malloc(X)
#include <stdint.h>
#define u64 uint64_t
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#define usleep_range(X,Y) sleep(X/1000)
#include <sys/time.h>
#include <sys/random.h>
u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
#define ktime_get_ns() get_tsns()
#include <stdbool.h>
#endif

#include "test_weights.h"
#include "helpers.h"
#include "predictors.h"
#include "variables.h"
#define FEAT_31
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

#define RUNS 1000
#define CPURUNS 5
bool check_correctness = false; 
#define CORRECTNESS_CHECKS 1000

u8 model_size = 0;

static char *cubin_path = "linnos.cubin";
static char *hsaco_path = "linnos.hsaco";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");
module_param(hsaco_path, charp, 0444);
MODULE_PARM_DESC(hsaco_path, "The path to linnos.hsaco, default ./linnos.hsaco");
#endif

long *test_weights[8] = { weight_0_T, weight_1_T, bias_0, bias_1, weight_M_1_T, bias_M_1, weight_M_2_T, bias_M_2};

char *apu_patterns[3] = {
    "linnos+0_APU_PL_batch_", "linnos+1_APU_PL_batch_", "linnos+2_APU_PL_batch_"
};

char *dgpu_patterns[3] = {
    "linnos+0_dGPU_batch_", "linnos+1_dGPU_batch_", "linnos+2_dGPU_batch_"
};

char *cpu_patterns[3]= {
    "linnos+0_CPU_batch_", "linnos+1_CPU_batch_", "linnos+2_CPU_batch_"
};
char out[1024];

static int run_apu(void) {

    //zerocpy pointer
    void *h_inputs_mapped = NULL;
    void *d_inputs_mapped = NULL;
    void *d_results_mapped = NULL;
    void *h_results_mapped = NULL;
    //zerocpy pointer

    int i, j;
    int batch_sizes[] = {16,1,2,4,8,16,32,64,128,256,512,1024};
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch_size = batch_sizes[n_batches-1];
    const int n = 1024;
    bool res;
    u64 false_count=0, true_count=0;
    u64 result_mismatches = 0;
    int batch_size;
    char input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times = NULL, *comp_run_times_1 = NULL, *comp_run_times_2 = NULL;
    u64* total_run_times = NULL;
    u64 avg, avg_1, avg_2, avg_total;
    u64 best, best_total;
    int nn;
    struct GPU_weights state;

    initialize_gpu(hsaco_path, max_batch_size);
    copy_weights(test_weights, &state);

    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    
    //生成一些垃圾输入，n是最大batchsize
	for(int b = 0 ; b < n; b++) 
		for(int j = 0; j < LEN_INPUT; j++)
			inputs_to_gpu[b*31 + j] =  (long) input[j];

    for (nn = 0 ; nn < 3 ; nn++) {
        // measuring GPU time
        for (i = 0 ; i < n_batches ; i++) {
            batch_size = batch_sizes[i];

            // copy inputs to GPU each time we run
            copy_inputs_to_gpu(batch_size);
            //warmup
            // if (nn==0) gpu_predict_batch_plus(0, batch_size, state.weights);
            // //else if(nn==1) gpu_predict_batch_plus_1(0, batch_size, state.weights);
            // //else  gpu_predict_batch_plus_2(0, batch_size, state.weights);
            // copy_results_from_gpu(batch_size);
            hipDeviceSynchronize();
//zerocpy setup memory--------------------------
                
            h_inputs_mapped = kava_alloc(sizeof(long) * LEN_INPUT * batch_size);
            hipHostRegister(h_inputs_mapped, sizeof(long) * LEN_INPUT * batch_size, hipHostRegisterMapped);
            hipHostGetDevicePointer(&d_inputs_mapped, h_inputs_mapped, 0); 
            long *linear_inputs = (long*)h_inputs_mapped;
            for(int b = 0 ; b < batch_size; b++) 
		        for(int j = 0; j < LEN_INPUT; j++)
			        linear_inputs[b*31 + j] =  (long) input[j];
            h_results_mapped = kava_alloc(sizeof(long) * 64 * batch_size);
            hipHostRegister(h_results_mapped, sizeof(long) * 64 * batch_size, hipHostRegisterMapped);
            hipHostGetDevicePointer(&d_results_mapped, h_results_mapped, 0); 
             
//zerocpy SET PARAMETERS---------------------------------
            void *args[] = {
                &state.weights[0], &state.weights[2], &d_inputs_mapped, &d_mid_res_i
            };
            void *args1[] = {
                &state.weights[1], &state.weights[3], &d_mid_res_i, &d_results_mapped
            };
            void *args2[] = {
                &state.weights[4], &state.weights[5], &d_mid_res_i, &d_mid_res_1_i
            };
            void *args3[] = {
                &state.weights[6], &state.weights[7], &d_mid_res_1_i, &d_mid_res_2_i
            };
            for (j = 0 ; j < RUNS ; j++) {
                PREDICT_GPU_SYNC = 1;
//ZEROCPY LAYER+0---------------------------------
                if (nn==0){
                c_start = ktime_get_ns();
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args, NULL),
                        "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            64, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args1, NULL),
                        "hipModuleLaunchKernel", __LINE__);
                if(PREDICT_GPU_SYNC == 1) {
                    check_error(hipDeviceSynchronize(), "hipDeviceSynchronize", __LINE__);}
                c_stop = ktime_get_ns();
                comp_run_times[j] = (c_stop - c_start);
                        }
//ZEROCPY LAYER+1---------------------------------
                        if (nn==1)
                        {
                c_start = ktime_get_ns();
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args, NULL),
                        "hipModuleLaunchKernel", __LINE__);
            
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_1_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args2, NULL),
                        "hipModuleLaunchKernel", __LINE__);
            
                check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            64, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args1, NULL),
                        "hipModuleLaunchKernel", __LINE__);
                if(PREDICT_GPU_SYNC == 1) {
                    check_error(hipDeviceSynchronize(), "hipDeviceSynchronize", __LINE__);
                }
                c_stop = ktime_get_ns();
                comp_run_times[j] = (c_stop - c_start);
                        }
//ZEROCPY LAYER+2---------------------------------       
                        if (nn==2){
                c_start = ktime_get_ns();      
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args, NULL),
                        "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_1_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args2, NULL),
                            "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_2_kernel, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args3, NULL),
                            "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel, 
                            batch_size, 1, 1,          //blocks
                            64, 1, 1,   //threads per block
                            0,   //shared mem
                            NULL, args1, NULL),
                            "hipModuleLaunchKernel", __LINE__);
                if(PREDICT_GPU_SYNC == 1) {
                            check_error(hipDeviceSynchronize(), "hipDeviceSynchronize", __LINE__);
                                }
                c_stop = ktime_get_ns();
                comp_run_times[j] = (c_stop - c_start);
                        }
            }

            avg = 0; 
            for (j = 0 ; j < RUNS ; j++) {
                avg += comp_run_times[j];
            }
            avg = avg / (1000*RUNS); 
            sprintf(out, "%s%d,%lld\n", apu_patterns[nn], batch_size, avg);
            PRINT("%s", out);
            
            // 释放循环内分配的资源
            if (h_results_mapped) {
                hipHostUnregister(h_results_mapped);
                kava_free(h_results_mapped);
                h_results_mapped = NULL;
            }
            if (h_inputs_mapped) {
                hipHostUnregister(h_inputs_mapped);
                kava_free(h_inputs_mapped);
                h_inputs_mapped = NULL;
            }
        }

    }


    for (nn = 0 ; nn < 3 ; nn++){
        // measuring cpu time
        for (i = 0 ; i < n_batches ; i++) {
            batch_size = batch_sizes[i];

            //warmup
            cpu_prediction_model_plus_2(input, 1, test_weights);
            if (nn==0) cpu_prediction_model(input, 1, test_weights);
            else if(nn==1) cpu_prediction_model_plus_1(input, 1, test_weights);
            else  cpu_prediction_model_plus_2(input, 1, test_weights);

            usleep_range(250, 1000);
        
            for (j = 0 ; j < CPURUNS ; j++) {
                t_start = ktime_get_ns();
                for(int k = 0; k < batch_size; k++) {
                    char input_copy[31];
                    memcpy (input_copy, input, sizeof(input));
                    if (nn==0) cpu_prediction_model(input, 1, test_weights);
                    else if(nn==1) cpu_prediction_model_plus_1(input, 1, test_weights);
                    else  cpu_prediction_model_plus_2(input, 1, test_weights);
                }
                t_stop = ktime_get_ns();
                
                //usleep_range(500, 2000);

                c_start = t_start;
                c_stop = t_stop;
                
               // usleep_range(500, 2000);
                comp_run_times[j] = (c_stop - c_start);
                total_run_times[j] = comp_run_times[j];//(t_stop - t_start);
            }

            avg = 0; avg_total = 0;
            best = 0; best_total = 0;
            for (j = 0 ; j < CPURUNS ; j++) {
                avg += comp_run_times[j];
                avg_total += total_run_times[j];
                if (best == 0 || comp_run_times[j] < best) best = comp_run_times[j];
                if (best_total == 0 || total_run_times[j] < best_total) best_total = total_run_times[j];
            }
            avg = avg / (1000*CPURUNS); avg_total = avg_total / (1000*CPURUNS);
            best = best / 1000; best_total = best_total / 1000;

            //PRINT("CPU_batch_%d,%lld,%lld,%lld,%lld\n", batch_size, avg, avg_total, best, best_total);
            sprintf(out, "%s%d,%lld\n", cpu_patterns[nn], batch_size, avg);
            PRINT("%s", out);
            //PRINT("linnos_CPU_batch_%d,%lld\n", batch_size, avg);
        }
    }

    if(check_correctness) {
        char *input_64 = kava_alloc(64 * LEN_INPUT * sizeof(char));
        for(int k = 0; k < CORRECTNESS_CHECKS; k++) {
            //generate random input
            #ifdef __KERNEL__ 
                get_random_bytes(input_64, 64 * LEN_INPUT);
            #else
                getrandom(input_64, 64 * LEN_INPUT, 0);
            #endif

            //the 1's here mean we only do 1 input, easy to adapt to n
            copy_input_to_shm(input_64, 64);
            copy_inputs_to_gpu(64);
            gpu_predict_batch(0, 64, state.weights);
            copy_results_from_gpu(64);
            
            for(int bnum = 0; bnum < 64; bnum++) {
                int cpu_result = cpu_prediction_model(input_64 + LEN_INPUT * bnum * sizeof(char), 1, test_weights);
                res = gpu_outputs[bnum*64]>=(gpu_outputs[bnum * 64 + 32])? false: true;
                //res = h_results_mapped[bnum*64]>=(h_results_mapped[bnum * 64 + 32])? false: true;
                //PRINT("Test [%d]: (%d) %s\n", bnum, res, res==cpu_result ? "Ok" : "WRONG");
                if (res!=cpu_result) result_mismatches++;
                if (cpu_result) true_count++;
                else false_count++;
            }            
        }
        PRINT("CPU prediction summary: %llu trues, %llu falses %llu result_mismatches\n", true_count, false_count, result_mismatches);
        // 释放 input_64
        if (input_64) {
            kava_free(input_64);
            input_64 = NULL;
        }
    }

    // 释放所有分配的资源
    gpu_cleanup(&state);
    if (comp_run_times) {
        vfree(comp_run_times);
        comp_run_times = NULL;
    }
    if (total_run_times) {
        vfree(total_run_times);
        total_run_times = NULL;
    }

    hipCtxDestroy(hipctx);
    return 0;
}

static int run_persistent(void) {

    //zerocpy pointer
    void *h_inputs_mapped = NULL;
    void *d_inputs_mapped = NULL;
    void *d_results_mapped = NULL;
    void *h_results_mapped = NULL;
    //zerocpy pointer

    int i, j;
    int batch_sizes[] = {16,1,2,4,8,16,32};
    //dont go beyound 32 or it might crash,still confusing why
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch_size = batch_sizes[n_batches-1];
    const int n = max_batch_size;
    bool res;
    u64 false_count=0, true_count=0;
    u64 result_mismatches = 0;
    int batch_size;
    char input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times, *comp_run_times_1, *comp_run_times_2;
    u64* total_run_times;
    u64 avg, avg_1, avg_2, avg_total;
    u64 best, best_total;
    int nn;
    struct GPU_weights state;

    initialize_gpu(hsaco_path, max_batch_size);
    copy_weights(test_weights, &state);

    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    if (!comp_run_times || !total_run_times) {
        PRINT("Failed to allocate memory for run times\n");
        return -1;
    }
 
//生成一些垃圾输入，n是最大batchsize
	for(int b = 0 ; b < n; b++) 
		for(int j = 0; j < LEN_INPUT; j++)
			inputs_to_gpu[b*31 + j] =  (long) input[j];
//Persistent Kernel Set Flags---------------------------------

 int* h_task_flag = (int*)kava_alloc(sizeof(int));
 int* h_quit_flag = (int*)kava_alloc(sizeof(int));
 if (!h_task_flag || !h_quit_flag) {
     PRINT("Failed to allocate memory for flags\n");
     vfree(comp_run_times);
     vfree(total_run_times);
     return -1;
 }
 if (hipHostRegister(h_task_flag, sizeof(int), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained) != hipSuccess) {
     PRINT("Failed to register h_task_flag\n");
     kava_free(h_task_flag);
     kava_free(h_quit_flag);
     vfree(comp_run_times);
     vfree(total_run_times);
     return -1;
 }
 if (hipHostRegister(h_quit_flag, sizeof(int), hipHostRegisterMapped) != hipSuccess) {
     PRINT("Failed to register h_quit_flag\n");
     hipHostUnregister(h_task_flag);
     kava_free(h_task_flag);
     kava_free(h_quit_flag);
     vfree(comp_run_times);
     vfree(total_run_times);
     return -1;
 }
 void* d_task_flag; 
 if (hipHostGetDevicePointer(&d_task_flag, h_task_flag, 0) != hipSuccess) {
     PRINT("Failed to get device pointer for d_task_flag\n");
     hipHostUnregister(h_task_flag);
     hipHostUnregister(h_quit_flag);
     kava_free(h_task_flag);
     kava_free(h_quit_flag);
     vfree(comp_run_times);
     vfree(total_run_times);
     return -1;
 }
 void* d_quit_flag; 
 if (hipHostGetDevicePointer(&d_quit_flag, h_quit_flag, 0) != hipSuccess) {
     PRINT("Failed to get device pointer for d_quit_flag\n");
     hipHostUnregister(h_task_flag);
     hipHostUnregister(h_quit_flag);
     kava_free(h_task_flag);
     kava_free(h_quit_flag);
     vfree(comp_run_times);
     vfree(total_run_times);
     return -1;
 }
//Persistent Kernel Set Streams---------------------------------

hipStream_t stream1, stream2, stream3, stream4;
#ifdef __KERNEL__
check_error(hipStreamCreate(&stream1,0), "hipStreamCreate stream1", __LINE__);
check_error(hipStreamCreate(&stream2,0), "hipStreamCreate stream2", __LINE__);
check_error(hipStreamCreate(&stream3,0), "hipStreamCreate stream3", __LINE__);
check_error(hipStreamCreate(&stream4,0), "hipStreamCreate stream4", __LINE__);
#else
check_error(hipStreamCreate(&stream1), "hipStreamCreate stream1", __LINE__);
check_error(hipStreamCreate(&stream2), "hipStreamCreate stream2", __LINE__);
check_error(hipStreamCreate(&stream3), "hipStreamCreate stream3", __LINE__);
check_error(hipStreamCreate(&stream4), "hipStreamCreate stream4", __LINE__);
#endif


    for (nn = 0 ; nn < 1 ; nn++) {
        // measuring GPU time
        for (i = 0 ; i < n_batches ; i++) {
            batch_size = batch_sizes[i];
            
            // 关键修复：在每次迭代开始时，先确保 persistent kernel 已停止
            // 然后才能安全地释放之前分配的内存
            if (i > 0) {
                // 停止之前的 persistent kernel（如果还在运行）
                *h_quit_flag = 1;
                __sync_synchronize();
                hipStreamSynchronize(stream1);
                hipStreamSynchronize(stream2);
                hipStreamSynchronize(stream3);
            }
            
            // 释放之前迭代分配的内存（如果存在）
            // 注意：必须在 persistent kernel 停止后才能释放
            if (h_results_mapped) {
                hipHostUnregister(h_results_mapped);
                kava_free(h_results_mapped);
                h_results_mapped = NULL;
            }
            if (h_inputs_mapped) {
                hipHostUnregister(h_inputs_mapped);
                kava_free(h_inputs_mapped);
                h_inputs_mapped = NULL;
            }
            
            //每个batch开始前，重置同步标志
            *h_task_flag = 0; *h_quit_flag = 0;
            __sync_synchronize();
            //zerocpy setup memory--------------------------

            h_inputs_mapped = kava_alloc(sizeof(long) * LEN_INPUT * batch_size);
            if (!h_inputs_mapped) {
                PRINT("Failed to allocate h_inputs_mapped for batch_size %d\n", batch_size);
                continue;
            }
            if (hipHostRegister(h_inputs_mapped, sizeof(long) * LEN_INPUT * batch_size, hipHostRegisterMapped) != hipSuccess) {
                PRINT("Failed to register h_inputs_mapped for batch_size %d\n", batch_size);
                kava_free(h_inputs_mapped);
                h_inputs_mapped = NULL;
                continue;
            }
            if (hipHostGetDevicePointer(&d_inputs_mapped, h_inputs_mapped, 0) != hipSuccess) {
                PRINT("Failed to get device pointer for d_inputs_mapped\n");
                hipHostUnregister(h_inputs_mapped);
                kava_free(h_inputs_mapped);
                h_inputs_mapped = NULL;
                continue;
            }
            long *linear_inputs = (long*)h_inputs_mapped;
            for(int b = 0 ; b < batch_size; b++) 
		        for(int j = 0; j < LEN_INPUT; j++)
			        linear_inputs[b*31 + j] =  (long) input[j];
            h_results_mapped = kava_alloc(sizeof(long) * 64 * batch_size);
            if (!h_results_mapped) {
                PRINT("Failed to allocate h_results_mapped for batch_size %d\n", batch_size);
                hipHostUnregister(h_inputs_mapped);
                kava_free(h_inputs_mapped);
                h_inputs_mapped = NULL;
                continue;
            }
            if (hipHostRegister(h_results_mapped, sizeof(long) * 64 * batch_size, hipHostRegisterMapped) != hipSuccess) {
                PRINT("Failed to register h_results_mapped for batch_size %d\n", batch_size);
                hipHostUnregister(h_inputs_mapped);
                kava_free(h_inputs_mapped);
                kava_free(h_results_mapped);
                h_inputs_mapped = NULL;
                h_results_mapped = NULL;
                continue;
            }
            if (hipHostGetDevicePointer(&d_results_mapped, h_results_mapped, 0) != hipSuccess) {
                PRINT("Failed to get device pointer for d_results_mapped\n");
                hipHostUnregister(h_inputs_mapped);
                hipHostUnregister(h_results_mapped);
                kava_free(h_inputs_mapped);
                kava_free(h_results_mapped);
                h_inputs_mapped = NULL;
                h_results_mapped = NULL;
                continue;
            }
//zerocpy SET PARAMETERS---------------------------------

            void *args[] = {
                &state.weights[0], &state.weights[2], &d_inputs_mapped, &d_mid_res_i, &d_task_flag, &d_quit_flag
            };
            void *args1[] = {
                &state.weights[1], &state.weights[3], &d_mid_res_i, &d_results_mapped, &d_task_flag, &d_quit_flag
            };
            void *args2[] = {
                &state.weights[4], &state.weights[5], &d_mid_res_i, &d_mid_res_1_i, &d_task_flag, &d_quit_flag
            };
            void *args3[] = {
                &state.weights[6], &state.weights[7], &d_mid_res_1_i, &d_mid_res_2_i, &d_task_flag, &d_quit_flag
            };

//Start Persistent Kernel LAYER+0---------------------------------
            // 关键修复：persistent kernel 只在第一次迭代时启动一次
            // 但每次迭代都会重新分配内存，所以需要确保：
            // 1. 第一次迭代时启动 persistent kernel
            // 2. 后续迭代时，persistent kernel 已经停止（通过 quit_flag），然后重新启动
            // 3. 或者使用固定的内存（不推荐，因为 batch_size 会变化）
            // 这里采用方案：每次迭代都重新启动 persistent kernel，确保使用当前的内存指针
            if (nn==0 && i==0){
                // 第一次启动 persistent kernel
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel_persistent, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            stream1, args, NULL),
                        "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel_persistent, 
                            batch_size, 1, 1,          //blocks
                            64, 1, 1,   //threads per block
                            0,   //shared mem
                            stream2, args1, NULL),
                        "hipModuleLaunchKernel", __LINE__);
            } else if (nn==0 && i>0) {
                // 后续迭代：重新启动 persistent kernel 使用新的内存指针
                // 注意：之前的 kernel 已经在循环开始时停止并同步
                check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel_persistent, 
                            batch_size, 1, 1,          //blocks
                            256, 1, 1,   //threads per block
                            0,   //shared mem
                            stream1, args, NULL),
                        "hipModuleLaunchKernel", __LINE__);

                check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel_persistent, 
                            batch_size, 1, 1,          //blocks
                            64, 1, 1,   //threads per block
                            0,   //shared mem
                            stream2, args1, NULL),
                        "hipModuleLaunchKernel", __LINE__);
            }
//Start Persistent Kernel LAYER+1---------------------------------
            // if (nn==1){
            //     check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_kernel_persistent, 
            //         batch_size, 1, 1,          //blocks
            //         256, 1, 1,   //threads per block
            //         0,   //shared mem
            //         stream1, args, NULL),
            //     "hipModuleLaunchKerne0", __LINE__);

            //     check_error(hipModuleLaunchKernel(batch_linnos_final_layer_kernel_persistent, 
            //         batch_size, 1, 1,          //blocks
            //         64, 1, 1,   //threads per block
            //         0,   //shared mem
            //         stream2, args1, NULL),
            //     "hipModuleLaunchKernel", __LINE__);

            //     check_error(hipModuleLaunchKernel(batch_linnos_mid_layer_1_kernel_persistent, 
            //         batch_size, 1, 1,          //blocks
            //         256, 1, 1,   //threads per block
            //         0,   //shared mem
            //         stream3, args2, NULL),
            //     "hipModuleLaunchKerne2", __LINE__);      
                   
            // }
            
            for (j = 0 ; j < 100; j++) {
                *h_task_flag = 1;
                while (*h_task_flag != 8) {
                    __sync_synchronize();  
                    }
                }

            for (j = 0 ; j < RUNS ; j++) {

                //测量persistent kernel
                c_start = ktime_get_ns();
                *h_task_flag = 1;
                while (*h_task_flag != 8) {
                    __sync_synchronize();   
                    //PRINT("h_task_flag_1 = %d,h_quit_flag = %d\n", *h_task_flag, *h_quit_flag);
                    }
                c_stop = ktime_get_ns();
                comp_run_times[j] = (c_stop - c_start);
                }
                
            *h_quit_flag = 1;
            __sync_synchronize();

            hipStreamSynchronize(stream1);
            hipStreamSynchronize(stream2);
            hipStreamSynchronize(stream3);
            hipStreamSynchronize(stream4);
            avg = 0; 

            for (j = 0 ; j < RUNS ; j++) {
                avg += comp_run_times[j];
            }
            avg = avg / (1000*RUNS); 
            //sprintf("_PK_%s%d,%lld\n", "linnos+0_APU_PK_batch_", batch_size, avg);
            PRINT("linnos+0_APU_PK_batch_%d,%lu\n", batch_size, avg);
            //PRINT("%s", out);
            
            // 在每次迭代结束时，确保persistent kernel已停止后再释放内存
            // 注意：这里不释放内存，因为persistent kernel可能还在使用
            // 内存将在下一次迭代开始时释放，或者在循环结束后释放
        }

    }
    
    // 在循环结束后，确保persistent kernel已停止，然后释放最后一次分配的内存
    *h_quit_flag = 1;
    __sync_synchronize();
    hipStreamSynchronize(stream1);
    hipStreamSynchronize(stream2);
    hipStreamSynchronize(stream3);
    hipStreamSynchronize(stream4);

    if (h_results_mapped) {
        hipHostUnregister(h_results_mapped);
        kava_free(h_results_mapped);
        h_results_mapped = NULL;
    }
    if (h_inputs_mapped) {
        hipHostUnregister(h_inputs_mapped);
        kava_free(h_inputs_mapped);
        h_inputs_mapped = NULL;
    }
    
    hipHostUnregister(h_task_flag);
    hipHostUnregister(h_quit_flag);
    kava_free(h_task_flag);
    kava_free(h_quit_flag);
    check_error(hipStreamDestroy(stream1), "hipStreamDestroy stream1", __LINE__);
    check_error(hipStreamDestroy(stream2), "hipStreamDestroy stream2", __LINE__);
    check_error(hipStreamDestroy(stream3), "hipStreamDestroy stream3", __LINE__);
    check_error(hipStreamDestroy(stream4), "hipStreamDestroy stream4", __LINE__);
    gpu_cleanup(&state);
    vfree(comp_run_times);
    vfree(total_run_times);
    hipCtxDestroy(hipctx);
    return 0;
    
}

static int run_dgpu(void) {
    int i, j;

    int batch_sizes[] = {16,1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch_size = batch_sizes[n_batches-1];
    // n needs to be at least as large as the largest batch size
    const int n = max_batch_size;
    bool res;
    u64 false_count=0, true_count=0;
    u64 result_mismatches = 0;
    int batch_size;
    char input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
    int nn;
    struct GPU_weights state;

    initialize_gpu_cuda(cubin_path, max_batch_size);
    copy_weights_cuda(test_weights, &state);

    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    expand_input_n_times(input, n);

    for (nn = 0 ; nn < 3 ; nn++) {
        // measuring GPU time
        for (i = 0 ; i < n_batches ; i++) {
            batch_size = batch_sizes[i];

            // // copy inputs to GPU each time we run
            // copy_inputs_to_gpu_cuda(batch_size);

            // // //warmup
            // if (nn==0) gpu_predict_batch_cuda(0, batch_size, state.weights);
            // else if(nn==1) gpu_predict_batch_plus_1_cuda(0, batch_size, state.weights);
            // else  gpu_predict_batch_plus_2_cuda(0, batch_size, state.weights);
            // copy_results_from_gpu_cuda(batch_size);
            // cuCtxSynchronize();
          
            for (j = 0 ; j < RUNS ; j++) {
                PREDICT_GPU_SYNC = 0;
                t_start = ktime_get_ns();
                copy_inputs_to_gpu_cuda(batch_size);
                if (nn==0) gpu_predict_batch_cuda(0, batch_size, state.weights);
                else if(nn==1) gpu_predict_batch_plus_1_cuda(0, batch_size, state.weights);
                else  gpu_predict_batch_plus_2_cuda(0, batch_size, state.weights);
                copy_results_from_gpu_cuda(batch_size);
                t_stop = ktime_get_ns();
                total_run_times[j] = (t_stop - t_start);
            }

            avg = 0; avg_total = 0;
            for (j = 0 ; j < RUNS ; j++) {
                avg += comp_run_times[j];
                avg_total += total_run_times[j];
            }
            avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
       
            sprintf(out, "%s%d,%lld\n", dgpu_patterns[nn], batch_size, avg_total);
            PRINT("%s", out);
        }
    }



    if(check_correctness) {
        char *input_64 = kava_alloc(64 * LEN_INPUT * sizeof(char));
        for(int k = 0; k < CORRECTNESS_CHECKS; k++) {
            //generate random input
            #ifdef __KERNEL__ 
                get_random_bytes(input_64, 64 * LEN_INPUT);
            #else
                getrandom(input_64, 64 * LEN_INPUT, 0);
            #endif

            //the 1's here mean we only do 1 input, easy to adapt to n
            copy_input_to_shm(input_64, 64);
            copy_inputs_to_gpu_cuda(64);
            gpu_predict_batch_cuda(0, 64, state.weights);
            copy_results_from_gpu_cuda(64);
            
            for(int bnum = 0; bnum < 64; bnum++) {
                int cpu_result = cpu_prediction_model(input_64 + LEN_INPUT * bnum * sizeof(char), 1, test_weights);
                res = gpu_outputs[bnum*64]>=(gpu_outputs[bnum * 64 + 32])? false: true;
                //PRINT("Test [%d]: (%d) %s\n", bnum, res, res==cpu_result ? "Ok" : "WRONG");
                if (res!=cpu_result) result_mismatches++;
                if (cpu_result) true_count++;
                else false_count++;
            }            
        }
        PRINT("CPU prediction summary: %llu trues, %llu falses %llu result_mismatches\n", true_count, false_count, result_mismatches);
    }

    gpu_cleanup_cuda(&state);
    vfree(comp_run_times);
    vfree(total_run_times);

    cuCtxDestroy(cuctx);
    return 0;
}


#ifdef __KERNEL__

/**
 * Program main
 */
static int __init linnos_init(void)
{   
    run_persistent();
    run_apu();
    run_dgpu();
	return 0;
}

static void __exit linnos_fini(void)
{

}

module_init(linnos_init);
module_exit(linnos_fini);

MODULE_AUTHOR("Henrique Fingler and Isha Tarte");
MODULE_DESCRIPTION("Kernel module of a linnos program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");

#else

int main() {
    run_apu();
    return 0;
}

#endif
