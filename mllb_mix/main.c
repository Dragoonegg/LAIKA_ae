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


#include "helpers.h"
#ifdef __KERNEL__
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#else
//if uspace
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>
#endif


#include "consts.h"

static char *hsaco_path = "mllb.hsaco";
#ifdef __KERNEL__
module_param(hsaco_path, charp, 0444);
MODULE_PARM_DESC(hsaco_path, "The path to mllb.hsaco, default ./mllb.hsaco");
#else
#endif

static char *cubin_path = "mllb.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to mllb.cubin, default ./mllb.cubin");
#else
#endif

static inline void check_malloc(void *p, const char* error_str, int line) {
    #ifdef __KERNEL__
    if (p == NULL) printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
    #else
    if (p == NULL) printf("ERROR: Failed to allocate %s (line %d)\n", error_str, line);
    #endif
}

struct matrix {
    int nrow;
    int ncol;
    float *values;
};
#define m2d(x, i, j) (x)->values[i * (x)->ncol + j]
#define m1d(x, i) (x)->values[i]
#define _ReLU(x) (x > 0 ?  x : 0)
__attribute__((target("sse")))
int matmul(struct matrix *X, struct matrix *Y, struct matrix *Z) 
{
    int i, j, k;
    for(i = 0; i < X->nrow; i++)
        for(j = 0; j < Y->ncol; j++)
            for(k = 0; k < X->ncol; k++) {
                m2d(Z, i, j) = m2d(Z, i, j) + (m2d(X, i, k) * m2d(Y, k, j));
            }
    return 0;
}
__attribute__((target("sse")))
void matadd(struct matrix *X, struct matrix *Y, struct matrix *Z)
{
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        Z->values[i] = X->values[i] + Y->values[i];
    }
}
__attribute__((target("sse")))
void ReLU(struct matrix *X)
{
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        X->values[i] = _ReLU(X->values[i]);
    }
}
__attribute__((target("sse")))
int forward_pass(struct matrix *input){
    float output;
    int ret;
    kernel_fpu_begin();
    float o1[10] = {0};
    float o2[10] = {0};
    struct matrix W1 = {NR_FEAT, 10, w1};
    struct matrix out1 = {1, 10, o1};
    struct matrix B1 = {1, 10, b1};
    struct matrix W2 = {10, 1, w2};
    struct matrix out2 = {1, 1, o2};
    struct matrix B2 = {1, 1, b2};
    matmul(input, &W1, &out1);
    matadd(&out1, &B1, &out1);
    ReLU(&out1);
    matmul(&out1, &W2, &out2);
    matadd(&out2, &B2, &out2);
    output = m1d(&out2, 0);
    //PRINT(V_INFO, "Output=%f\n", output);
    ret = output > 0.5 ? 1 : 0;
    kernel_fpu_end();
    return ret;
}

static int run_cpu(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    int i, j, k;
    int *tmp;
    int batch_size;
    int rand_counter = 0;
    u64 t_start, t_stop;
    u64* total_run_times;
    u64 avg;
    
    struct matrix *inputs = (struct matrix*) vmalloc(max_batch*sizeof(struct matrix));
    for (j = 0 ; j < max_batch ; j++) {
        inputs[j].values = (float*) vmalloc(NR_FEAT*sizeof(float));
        inputs[j].nrow = 1;
        inputs[j].ncol = NR_FEAT;
        for (i = 0 ; i < NR_FEAT ; i++) {
            tmp = (int*) inputs[j].values+i;
            *tmp = rand_floats_as_int[rand_counter];
            rand_counter++;
            if (rand_counter == 4) rand_counter = 0;
        }
    }

    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        //warmup
        forward_pass(inputs);
        usleep_range(250, 1000);

        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            for (k = 0 ; k < batch_size ; k++) {
                // // 打印 inputs+k 的内容
                // PRINT(V_INFO, "Inputs[%d] 矩阵内容 (nrow=%d, ncol=%d): ", k, inputs[k].nrow, inputs[k].ncol);
                // for (int feat = 0; feat < NR_FEAT; feat++) {
                //     PRINT(V_INFO, "%.6f ", inputs[k].values[feat]);
                // }
                // PRINT(V_INFO, "\n");
                
                forward_pass(inputs+k);
                //PRINT(V_INFO, "Output[%d]=%d\n", k, forward_pass(inputs+k));
            }
            t_stop = ktime_get_ns();
            total_run_times[j] = (t_stop - t_start);
            usleep_range(250, 1000);

        }

        avg = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += total_run_times[j];
        }
        //printf("sum before division: %lu\n", avg);
        avg = avg / (RUNS*1000);
      //  printf("avg after division: %lu\n", avg);
        PRINT(V_INFO, "MLLB_CPU_batch_%d, %lu\n", batch_size,  avg);
    }

    for (j = 0 ; j < max_batch ; j++) {
        vfree(inputs[j].values);
    }
    vfree(inputs);
    return 0;
}


static int run_dgpu(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    int i, j;
    int* linear_inputs;
    int batch_size;
    int rand_counter = 0;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
    float* outs;

    //init cuda context
    CUcontext cuContext;
    CUfunction batch_mllb_kernel;
    CUdeviceptr d_inputs, d_w1, d_b1, d_w2, d_results;
    gpu_init_cuda(0, &cuContext);

    linear_inputs = kava_alloc(NR_FEAT*max_batch*sizeof(float));
    check_malloc(linear_inputs, "check_malloc", __LINE__);

    //initialize a linear matrix with fake inputs
    for (j = 0 ; j < max_batch*NR_FEAT ; j++) {
        linear_inputs[j] = rand_floats_as_int[rand_counter];
        rand_counter++;
        if (rand_counter == 4) rand_counter = 0;
    }

    gpu_get_cufunc_cuda(cubin_path, "_Z18mllb_infer_v2_cudaPfS_S_S_fS_", &batch_mllb_kernel);
    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        gpu_setup_cuda(batch_size, &d_inputs, &d_w1, &d_b1, &d_w2, &d_results);
        outs = kava_alloc(batch_size * sizeof(float));

        //warmup
        gpu_setup_inputs_cuda(d_inputs, linear_inputs, batch_size);
        gpu_inference_many_cuda(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results, 0);
        cuCtxSynchronize();
        gpu_inference_many_cuda(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results, 0);
        cuCtxSynchronize();
        usleep_range(1000, 2000);

        //do the entire algorithm
        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            gpu_setup_inputs_cuda(d_inputs, linear_inputs, batch_size);
            gpu_inference_many_cuda(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results, 1);
            gpu_get_result_cuda(batch_size, d_results, outs);
            t_stop = ktime_get_ns();

            total_run_times[j] = (t_stop - t_start);
            //usleep_range(1000, 2000);
        }

     

        avg = 0; avg_total = 0;
      
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
        }
        avg = avg / (1000*RUNS); 
        avg_total = avg_total / (1000*RUNS);
     

        //PRINT(V_INFO, "GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
        PRINT(V_INFO, "MLLB_dGPU_batch_%d, "LLD"\n", batch_size, avg_total);
        gpu_clean_cuda(d_inputs, d_w1, d_b1, d_w2, d_results);
        kava_free(outs);

    }

    kava_free(linear_inputs);
    vfree(comp_run_times);
    vfree(total_run_times);
    return 0;
}


static int run_apu_zerocopy(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    int i, j;
    int* linear_inputs;
    int batch_size;
    int rand_counter = 0;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
    float* outs;

    //init hip context
    hipCtx_t cuctx;
    hipFunction_t batch_mllb_kernel;
    void *d_inputs, *d_w1, *d_b1, *d_w2, *d_results;
    void *h_inputs_mapped; // 新增：映射的主机内存
    void *d_inputs_mapped; // 新增：对应的设备指针
    // --- zerocopy 结果输出 ---
    void *h_results_mapped;
    void *d_results_mapped;
    gpu_init(0, &cuctx);
    
    h_inputs_mapped = kava_alloc(NR_FEAT*max_batch*sizeof(float));
    hipHostRegister(h_inputs_mapped, NR_FEAT*max_batch*sizeof(float),hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
    hipHostGetDevicePointer(&d_inputs_mapped, h_inputs_mapped, 0);

    h_results_mapped = kava_alloc(max_batch * sizeof(float));
    hipHostRegister(h_results_mapped, max_batch * sizeof(float),hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
    hipHostGetDevicePointer(&d_results_mapped, h_results_mapped, 0);
        
    linear_inputs = (int*)h_inputs_mapped; // 使用映射的内存
    // PRINT(V_INFO, "=== 内存映射指针值 ===\n");
    // PRINT(V_INFO, "h_inputs_mapped (主机指针): %p\n", h_inputs_mapped);
    // PRINT(V_INFO, "d_inputs_mapped (设备指针): %p\n", d_inputs_mapped);
    // PRINT(V_INFO, "linear_inputs (int* 视图): %p\n", linear_inputs);
    // PRINT(V_INFO, "内存大小: %lu 字节 (%d * %d * %lu)\n", 
    //        (unsigned long)(NR_FEAT * max_batch * sizeof(float)), 
    //        NR_FEAT, max_batch, sizeof(float));
    // PRINT(V_INFO, "=== 指针值打印完成 ===\n\n");
    //创建一个线性矩阵，并填充随机数
    for (j = 0 ; j < max_batch*NR_FEAT ; j++) {
        linear_inputs[j] = rand_floats_as_int[rand_counter];
        //PRINT(V_INFO, "linear_inputs[%d]=%d\n", j, linear_inputs[j]);
        rand_counter++;
        if (rand_counter == 4) rand_counter = 0;
    }

    gpu_get_cufunc(hsaco_path, "_Z13mllb_infer_v2PfS_S_S_fS_", &batch_mllb_kernel);
    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

        
 

    for (i = 0 ; i < n_batches ; i++) {
        //对每个batch大小，获取一个batch大小的d_results，并设置输入矩阵，并拷贝到GPU
        batch_size = batch_sizes[i];
        gpu_setup_zerocopy(batch_size, &d_w1, &d_b1, &d_w2);
        
        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();  
            int threadsPerBlock = 128;
            int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;  // 根据batch_size计算blocks
            if (blocks == 0) blocks = 1;
            void* args[] = { &d_inputs_mapped, &d_w1, &d_b1, &d_w2, &b2, &d_results_mapped}; // 使用映射的设备指针
            check_error(hipModuleLaunchKernel(batch_mllb_kernel, 
                blocks, 1, 1,      //blocks
                threadsPerBlock, 1, 1,          //threads per block
                0, //shared mem - 不需要共享内存
                NULL, args, NULL),
            "hipModuleLaunchKernel", __LINE__);
            hipDeviceSynchronize();
            t_stop = ktime_get_ns();
            total_run_times[j] = (t_stop - t_start);
        }
        
        // //do just computation
        // for (j = 0 ; j < RUNS; j++) {
        //     int threadsPerBlock = 128;
        //     int blocks = (batch_size + threadsPerBlock - 1) / threadsPerBlock;  // 根据batch_size计算blocks
        //     if (blocks == 0) blocks = 1;

        //     void* args[] = { &d_inputs_mapped, &d_w1, &d_b1, &d_w2, &b2, &d_results_mapped }; // 使用映射的设备指针
            
        //     c_start = ktime_get_ns();
        //     check_error(hipModuleLaunchKernel(batch_mllb_kernel, 
		// 		blocks, 1, 1,      //blocks
		// 		threadsPerBlock, 1, 1,          //threads per block
		// 		0, //shared mem - 不需要共享内存
        //         NULL, args, NULL),
		// 	"hipModuleLaunchKernel", __LINE__);
        //     hipDeviceSynchronize();
        //     c_stop = ktime_get_ns();
            
        //     float ms = 0;
        //     comp_run_times[j] = (c_stop - c_start);
          
        // }

        avg = 0; avg_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg_total += total_run_times[j];
           
        }
        avg = avg / (1000*RUNS);  // 将纳秒转换为微秒
        avg_total = avg_total / (1000*RUNS);  // 将纳秒转换为微秒
  
        PRINT(V_INFO, "MLLB_APU_PL_batch_%d, %lu\n", batch_size,  avg_total);
        //  if (h_results_mapped != NULL && batch_size > 0) {
        //     float* results_view = (float*)h_results_mapped;
        //     float sum = 0.0f;
            
        //     // 将所有元素相加
        //     for (int pi = 0; pi < batch_size; ++pi) {
        //         sum += results_view[pi];
        //     }
            
        //     PRINT(V_INFO, "[Batch %d] h_results_mapped 所有元素总和: %.6f\n", batch_size, sum);
        // } else {
        //     PRINT(V_INFO, "[Batch %d] d_result allocation failed or batch_size invalid\n", batch_size);
        // }

        gpu_clean(NULL, d_w1, d_b1, d_w2, NULL); 
        
    }
    
    

    // 释放映射的内存
    hipHostUnregister(h_inputs_mapped);
    hipHostUnregister(h_results_mapped);
    kava_free(h_inputs_mapped);
    kava_free(h_results_mapped);
    
    vfree(comp_run_times);
    vfree(total_run_times);
    
    return 0;
}






static int run_apu_persistent(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    
    hipCtx_t cuctx;
    hipFunction_t persistent_kernel;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64* flag_set_times;  
    u64* flag_wait_times; 
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    flag_set_times = (u64*) vmalloc(RUNS*sizeof(u64));
    flag_wait_times = (u64*) vmalloc(RUNS*sizeof(u64));
    int avg_total = 0;

    gpu_init(0, &cuctx);
    //gpu_get_cufunc(hsaco_path, "_Z27mllb_persistent_infer_2_optPfS_S_S_fS_PiS0_S0_", &persistent_kernel);
    gpu_get_cufunc(hsaco_path, "_Z21mllb_persistent_inferPfS_S_S_fS_PiS0_S0_", &persistent_kernel);
    void* h_inputs = kava_alloc(NR_FEAT * max_batch * sizeof(float));
    float* h_results = (float*)kava_alloc(max_batch * sizeof(float));
    int* h_task_flag = (int*)kava_alloc(sizeof(int));
    int* h_quit_flag = (int*)kava_alloc(sizeof(int));
    int* h_batch_size = (int*)kava_alloc(sizeof(int));
    *h_task_flag = 0; *h_quit_flag = 0; *h_batch_size = 0;
    hipHostRegister(h_inputs, NR_FEAT*max_batch*sizeof(float), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
    hipHostRegister(h_results, max_batch*sizeof(float), hipHostRegisterMapped);
    hipHostRegister(h_task_flag, sizeof(int), hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
    hipHostRegister(h_quit_flag, sizeof(int), hipHostRegisterMapped);
    hipHostRegister(h_batch_size, sizeof(int), hipHostRegisterMapped);

    void* d_inputs; hipHostGetDevicePointer(&d_inputs, h_inputs, 0);
    void* d_results; hipHostGetDevicePointer(&d_results, h_results, 0);
    void* d_task_flag; hipHostGetDevicePointer(&d_task_flag, h_task_flag, 0);
    void* d_quit_flag; hipHostGetDevicePointer(&d_quit_flag, h_quit_flag, 0);
    void* d_batch_size; hipHostGetDevicePointer(&d_batch_size, h_batch_size, 0);

    

    void *d_w1, *d_b1, *d_w2;
    gpu_setup_zerocopy(max_batch, &d_w1, &d_b1, &d_w2);
 
        for (int z = 0; z < n_batches&&batch_sizes[z]<=4096; z++)
        {
        *h_task_flag = 0; *h_quit_flag = 0; *h_batch_size = 0;
         // 启动persistent kernel
        hipStream_t stream1;
        #ifdef __KERNEL__
        check_error(hipStreamCreate(&stream1,0), "hipStreamCreate1", __LINE__);
        #else
        check_error(hipStreamCreate(&stream1), "hipStreamCreate1", __LINE__);
        #endif
        void* args[] = { &d_inputs, &d_w1, &d_b1, &d_w2, &b2, &d_results, &d_task_flag, &d_quit_flag, &d_batch_size };
        int threadsPerBlock = 256, blocks = 1;  
        check_error(hipModuleLaunchKernel(persistent_kernel, blocks, 1, 1, threadsPerBlock, 1, 1, 0, stream1, args, NULL), "hipModuleLaunchKernel", __LINE__);
        int rand_counter = 0;
        int batch_size = batch_sizes[z];
        *h_batch_size = batch_size;
        #ifdef __KERNEL__
        mb();
        #else
        __sync_synchronize();
        #endif
            int* linear_inputs = (int*)h_inputs;
            for (int j = 0 ; j < batch_size*NR_FEAT ; j++) {
                linear_inputs[j] = rand_floats_as_int[rand_counter];
                //PRINT(V_INFO, "linear_inputs[%d]=%d\n", j, linear_inputs[j]);
                rand_counter++;
                if (rand_counter == 4) rand_counter = 0;
            }

    // 添加内存屏障确保数据一致性
    #ifdef __KERNEL__
    mb();
    #else
    __sync_synchronize();
    #endif

    //热身
    for (int j = 0 ; j < 10000; j++) { 
        *h_task_flag = 1;
        while (*h_task_flag == 1) {}
    }
        for (int j = 0 ; j < RUNS ; j++) { 
        t_start = ktime_get_ns();  
        *h_task_flag = 1;
        while (*h_task_flag == 1) {}
        t_stop = ktime_get_ns();
        total_run_times[j] = t_stop-t_start; // 只计算等待时间
    }
    
 
    *h_quit_flag = 1;
    hipStreamSynchronize(stream1);
    check_error(hipStreamDestroy(stream1), "hipStreamDestroy stream1", __LINE__);


    for (int j = 0 ; j < RUNS ; j++) {
        avg_total += total_run_times[j];
        }
    avg_total = avg_total / (1000*RUNS);  // 将纳秒转换为微秒
    PRINT(V_INFO, "MLLB_APU_PK_batch_%d,%lu\n", batch_size, avg_total);
    }
        
    // 释放资源
    hipHostUnregister(h_inputs); 
    hipHostUnregister(h_results);
    hipHostUnregister(h_task_flag); 
    hipHostUnregister(h_quit_flag); 
    hipHostUnregister(h_batch_size);
    kava_free(h_inputs); kava_free(h_results); kava_free(h_task_flag); kava_free(h_quit_flag); kava_free(h_batch_size);
    vfree(flag_set_times); vfree(flag_wait_times); vfree(total_run_times);
    gpu_clean(NULL, d_w1, d_b1, d_w2, NULL);
    return 0;
}


static int run(void) {
    int batch_sizes[] = {16,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536};
    //int batch_sizes[] = {1024,2048,4096};
    int n_batches = sizeof(batch_sizes)/sizeof(int);
    int max_batch = batch_sizes[n_batches-1];
    int RUNS = 10000;
    int CPURUNS=10;
    int rand_floats_as_int[] = {1036831949, 1045220557, 1050253722, -1110651699};
    
    run_cpu(batch_sizes, n_batches, max_batch, CPURUNS, rand_floats_as_int);
    run_dgpu(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);
    run_apu_zerocopy(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);
    //run_gpu_memcpy(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);
    run_apu_persistent(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);

    
    return 0;
}


#ifdef __KERNEL__

static int __init mllb_init(void)
{
	return run();
}

static void __exit mllb_fini(void)
{
    //cleanup
}

module_init(mllb_init);
module_exit(mllb_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_AUTHOR("Haoming Zhuo");
MODULE_DESCRIPTION( "Adapted kernel module based on the original MLLB implementation by Henrique Fingler");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0.0");
#else

int main() {
    run();
    return 0;
}

#endif