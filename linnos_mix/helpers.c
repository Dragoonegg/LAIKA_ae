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
#include "variables.h"
#include "helpers.h"
#include "predictors.h"


static void gpu_init(int dev) {
    int cuDevice;
    hipError_t res;

    hipInit(0);
    res = hipDeviceGet(&cuDevice, dev);
    if (res != hipSuccess){
        PRINT("cannot acquire apu device 0\n");
    }

    res = hipCtxCreate(&hipctx, 0, cuDevice);
    if (res != hipSuccess){
        PRINT("cannot create context\n");
    }
}

static void gpu_init_cuda(int dev) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        PRINT("cannot acquire cuda device 0\n");
    }

    res = cuCtxCreate(&cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        PRINT("cannot create context\n");
    }
}

static void gpu_get_cufunc(const char* cubin, char* kname, hipFunction_t *func) {
    hipModule_t hipModule;
    hipError_t res;
    res = hipModuleLoad(&hipModule, cubin);
    if (res != hipSuccess) {
        PRINT("cannot load module: %d\n", res);
    }   

    res = hipModuleGetFunction(func, hipModule, kname);
    if (res != hipSuccess){
        PRINT("cannot acquire kernel handle\n");
    }   
}

static void gpu_get_cufunc_cuda(const char* cubin, char* kname, CUfunction *func) {
    CUmodule cuModule;
    CUresult res;
    res = cuModuleLoad(&cuModule, cubin);
    if (res != CUDA_SUCCESS) {
        PRINT("cannot load module: %d\n", res);
    }

    res = cuModuleGetFunction(func, cuModule, kname);
    if (res != CUDA_SUCCESS){
        PRINT("cannot acquire kernel handle\n");
    }
}


//this is multi ssd ready
void copy_weights(long **weights, struct GPU_weights *state) {
    long *kbuf_weight_0_T_ent;
    long *kbuf_weight_1_T_ent;
    long *kbuf_weight_M_1;
    long *kbuf_weight_M_2;
    long *kbuf_bias_0_ent;
    long *kbuf_bias_1_ent;
    long *kbuf_bias_M_1;
    long *kbuf_bias_M_2;

	check_error(hipMalloc((void**) &state->weights[0], sizeof(long) * 256*31), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[1], sizeof(long) * 256*2), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[2], sizeof(long) * 256), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[3], sizeof(long) * 2), "hipMalloc ", __LINE__);

    check_error(hipMalloc((void**) &state->weights[4], sizeof(long) * 256 * 256), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[5], sizeof(long) * 256), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[6], sizeof(long) * 256 * 256), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &state->weights[7], sizeof(long) * 256), "hipMalloc ", __LINE__);


    //initialize variables
	kbuf_weight_0_T_ent = (long*) kava_alloc(256*31*sizeof(long));
    memcpy(kbuf_weight_0_T_ent, weights[0], 256*31*sizeof(long));
    kbuf_bias_0_ent = (long*) kava_alloc(256*sizeof(long));
    memcpy(kbuf_bias_0_ent, weights[2], 256*sizeof(long));

    kbuf_weight_1_T_ent = (long*) kava_alloc(256*2*sizeof(long));
    memcpy(kbuf_weight_1_T_ent, weights[1], 256*2*sizeof(long));
    kbuf_bias_1_ent = (long*) kava_alloc(2*sizeof(long));
    memcpy(kbuf_bias_1_ent, weights[3], 2*sizeof(long));

    check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[0], kbuf_weight_0_T_ent, sizeof(long) * 256*31), "hipMemcpyHtoD", __LINE__);
	check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[1], kbuf_weight_1_T_ent, sizeof(long) * 256*2), "hipMemcpyHtoD", __LINE__);
	check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[2], kbuf_bias_0_ent, sizeof(long) * 256), "hipMemcpyHtoD", __LINE__);
	check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[3], kbuf_bias_1_ent, sizeof(long) * 2), "hipMemcpyHtoD", __LINE__);

    kava_free(kbuf_weight_0_T_ent);
    kava_free(kbuf_weight_1_T_ent);
    kava_free(kbuf_bias_0_ent);
    kava_free(kbuf_bias_1_ent);

    //test if +1
    if (weights[4] && weights[5]) {
        //pr_warn("Copying weights for +1\n");
        kbuf_weight_M_1 = (long*) kava_alloc(256*256*sizeof(long));
        memcpy(kbuf_weight_M_1, weights[4], 256*256*sizeof(long));

        kbuf_bias_M_1 = (long*) kava_alloc(256*sizeof(long));
        memcpy(kbuf_bias_M_1, weights[5], 256*sizeof(long));
        
        check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[4], kbuf_weight_M_1, sizeof(long) * 256 * 256), "hipMemcpyHtoD", __LINE__);
        check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[5], kbuf_bias_M_1, sizeof(long) * 256), "hipMemcpyHtoD", __LINE__);
        kava_free(kbuf_weight_M_1);
        kava_free(kbuf_bias_M_1);
    }
    
    //test if +2
    if (weights[6] && weights[7]) {
        //pr_warn("Copying weights for +2\n");
        kbuf_weight_M_2 = (long*) kava_alloc(256*256*sizeof(long));
        memcpy(kbuf_weight_M_2, weights[6], 256*256*sizeof(long));

        kbuf_bias_M_2 = (long*) kava_alloc(256*sizeof(long));
        memcpy(kbuf_bias_M_2, weights[7], 256*sizeof(long));
        
        check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[6], kbuf_weight_M_2, sizeof(long) * 256 * 256), "hipMemcpyHtoD", __LINE__);
        check_error(hipMemcpyHtoD((hipDeviceptr_t )state->weights[7], kbuf_bias_M_2, sizeof(long) * 256), "hipMemcpyHtoD", __LINE__);
        kava_free(kbuf_weight_M_2);
        kava_free(kbuf_bias_M_2);
    }
}

void copy_results_from_gpu(u64 n_inputs) {
    hipMemcpyDtoH(gpu_outputs, d_final_res_i, sizeof(long) * 64 * n_inputs);
}

void initialize_gpu(const char* hsaco_path, int max_batch_size) {
    //intialize kernels
    // if (hipctx) {
    //     return;
    // }
    gpu_init(0);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(hsaco_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_mid_layer_1_batchPlS_S_S_", &batch_linnos_mid_layer_1_kernel);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_mid_layer_2_batchPlS_S_S_", &batch_linnos_mid_layer_2_kernel);

    gpu_get_cufunc(hsaco_path, "_Z39prediction_final_layer_batch_persistentPlS_S_S_PiS0_", &batch_linnos_final_layer_kernel_persistent);
    gpu_get_cufunc(hsaco_path, "_Z37prediction_mid_layer_batch_persistentPlS_S_S_PiS0_", &batch_linnos_mid_layer_kernel_persistent);

    check_error(hipMalloc((void**) &d_input_vec_i, sizeof(long) * LEN_INPUT * max_batch_size), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &d_mid_res_i,   sizeof(long) * LEN_LAYER_0 * max_batch_size), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &d_mid_res_1_i, sizeof(long) * LEN_LAYER_M_1 * max_batch_size), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &d_mid_res_2_i, sizeof(long) * LEN_LAYER_M_2 * max_batch_size), "hipMalloc ", __LINE__);
    check_error(hipMalloc((void**) &d_final_res_i, sizeof(long) * LEN_LAYER_1 * max_batch_size *32), "hipMalloc ", __LINE__);

    inputs_to_gpu = kava_alloc(LEN_INPUT * max_batch_size * sizeof(long));
    if (!inputs_to_gpu) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
    gpu_outputs = kava_alloc(64 * max_batch_size * sizeof(long));
    if (!gpu_outputs) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
}

void initialize_gpu_cuda(const char* cubin_path, int max_batch_size) {
    //intialize kernels
    if (cuctx) {
        return;
    }
    gpu_init_cuda(0);
    gpu_get_cufunc_cuda(cubin_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel_cuda);
    gpu_get_cufunc_cuda(cubin_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel_cuda);
    gpu_get_cufunc_cuda(cubin_path, "_Z28prediction_mid_layer_1_batchPlS_S_S_", &batch_linnos_mid_layer_1_kernel_cuda);
    gpu_get_cufunc_cuda(cubin_path, "_Z28prediction_mid_layer_2_batchPlS_S_S_", &batch_linnos_mid_layer_2_kernel_cuda);

    check_error(cuMemAlloc((CUdeviceptr*) &d_input_vec_i_cuda, sizeof(long) * LEN_INPUT * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_i_cuda,   sizeof(long) * LEN_LAYER_0 * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_1_i_cuda, sizeof(long) * LEN_LAYER_M_1 * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_2_i_cuda, sizeof(long) * LEN_LAYER_M_2 * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_final_res_i_cuda, sizeof(long) * LEN_LAYER_1 * max_batch_size *32), "cuMemAlloc ", __LINE__);

    inputs_to_gpu = kava_alloc(LEN_INPUT * max_batch_size * sizeof(long));
    if (!inputs_to_gpu) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
    gpu_outputs = kava_alloc(64 * max_batch_size * sizeof(long));
    if (!gpu_outputs) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
}


void gpu_cleanup(struct GPU_weights *state) {
    int i;
    //pr_warn("Cleaning up APU state\n");
    for(i = 0; i <8 ; i++) {
        hipFree((hipDeviceptr_t)state->weights[i]);
    }
    hipFree(d_input_vec_i);
    hipFree(d_mid_res_i);
    hipFree(d_mid_res_1_i);
    hipFree(d_mid_res_2_i);
    hipFree(d_final_res_i);

    if (!inputs_to_gpu) {
        kava_free(inputs_to_gpu);
        inputs_to_gpu = 0;
    }
    if (!gpu_outputs) {
        kava_free(gpu_outputs);
        gpu_outputs = 0;
    }
}

void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) PRINT("ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

//this takes one input array (with LEN_INPUT bytes) and
//expands it to N inputs, while converting to longs
void expand_input_n_times(char* input, int n) {
    int b, j;
	for(b = 0 ; b < n; b++) 
		for(j = 0; j < LEN_INPUT; j++)
			inputs_to_gpu[b*31 + j] =  (long) input[j];
}

void copy_input_to_shm(char* input, int n) {
    int j;
    for(j = 0; j < n * LEN_INPUT; j++)
	    inputs_to_gpu[j] =  (long) input[j];
}

//pass number of inputs, not bytes
void copy_inputs_to_gpu(u64 n_inputs) {
    hipMemcpyHtoD(d_input_vec_i, inputs_to_gpu, sizeof(long) * LEN_INPUT * n_inputs);
}

/*
 * Multi GPU, multi batch functions
*/

void multi_gpu_cleanup_dev(struct GPU_weights *state, int dev) {
    int i, batch;
    //pr_warn("Cleaning up GPU %d state\n", dev);
    for(i = 0; i < 8 ; i++) {
        hipFree((hipDeviceptr_t)state->weights[i]);
    }

    for(batch = 0 ; batch < MAX_DEV_BATCHES ; batch++){
        //pr_warn("Freeing for %d/%d\n", dev, batch);
        hipFree(multi_d_input_vec_i[dev][batch]);
        hipFree(multi_d_mid_res_i[dev][batch]);
        hipFree(multi_d_mid_res_1_i[dev][batch]);
        hipFree(multi_d_mid_res_2_i[dev][batch]);
        hipFree(multi_d_final_res_i[dev][batch]);

        kava_free(multi_inputs_to_gpu[dev][batch]);        
        kava_free(multi_gpu_outputs[dev][batch]);

        hipStreamDestroy(cu_streams[dev][batch]);
    }
}

void multi_initialize_gpu(const char* hsaco_path, int max_batch_size, int ndev) {
    int dev, batch;
    //intialize kernels
    if (cuctx) 
        return;

    gpu_init(0);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(hsaco_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_mid_layer_1_batchPlS_S_S_", &batch_linnos_mid_layer_1_kernel);
    gpu_get_cufunc(hsaco_path, "_Z28prediction_mid_layer_2_batchPlS_S_S_", &batch_linnos_mid_layer_2_kernel);

    gpu_get_cufunc(hsaco_path, "_Z39prediction_final_layer_batch_persistentPlS_S_S_PiS0_", &batch_linnos_final_layer_kernel_persistent);
    gpu_get_cufunc(hsaco_path, "_Z37prediction_mid_layer_batch_persistentPlS_S_S_PiS0_", &batch_linnos_mid_layer_kernel_persistent);
    
    for(dev = 0 ; dev < ndev ; dev++){
        for(batch = 0 ; batch < MAX_DEV_BATCHES ; batch++){
            check_error(hipMalloc((void**) &multi_d_input_vec_i[dev][batch], sizeof(long) * LEN_INPUT * max_batch_size), "hipMalloc ", __LINE__);
            check_error(hipMalloc((void**) &multi_d_mid_res_i[dev][batch], sizeof(long) * LEN_LAYER_0 * max_batch_size), "hipMalloc ", __LINE__);
            check_error(hipMalloc((void**) &multi_d_mid_res_1_i[dev][batch], sizeof(long) * LEN_LAYER_M_1 * max_batch_size), "hipMalloc ", __LINE__);
            check_error(hipMalloc((void**) &multi_d_mid_res_2_i[dev][batch], sizeof(long) * LEN_LAYER_M_2 * max_batch_size), "hipMalloc ", __LINE__);
            check_error(hipMalloc((void**) &multi_d_final_res_i[dev][batch], sizeof(long) * LEN_LAYER_1 * max_batch_size *32), "hipMalloc ", __LINE__);

            check_error(hipStreamCreate(&cu_streams[dev][batch], 0), "hipMalloc ", __LINE__);

            multi_inputs_to_gpu[dev][batch] = kava_alloc(LEN_INPUT * max_batch_size * sizeof(long));
            if (!multi_inputs_to_gpu[dev][batch]) 
                pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
            
            multi_gpu_outputs[dev][batch] = kava_alloc(64 * max_batch_size * sizeof(long));
            if (!multi_gpu_outputs[dev][batch]) 
                pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
        }
    }
}

void multi_copy_inputs_to_gpu(u64 n_inputs, int dev, int batch_id) {
    hipMemcpyHtoDAsync(multi_d_input_vec_i[dev][batch_id], multi_inputs_to_gpu[dev][batch_id], sizeof(long) * LEN_INPUT * n_inputs, cu_streams[dev][batch_id]);
}

void multi_copy_results_from_gpu(u64 n_inputs, int dev, int batch_id) {
    hipMemcpyDtoHAsync(multi_gpu_outputs[dev][batch_id], 
            multi_d_final_res_i[dev][batch_id], 
            sizeof(long) * 64 * n_inputs, 
            cu_streams[dev][batch_id]);
    hipStreamSynchronize(cu_streams[dev][batch_id]);
}


void copy_weights_cuda(long **weights, struct GPU_weights *state) {
    long *kbuf_weight_0_T_ent;
    long *kbuf_weight_1_T_ent;
    long *kbuf_weight_M_1;
    long *kbuf_weight_M_2;
    long *kbuf_bias_0_ent;
    long *kbuf_bias_1_ent;
    long *kbuf_bias_M_1;
    long *kbuf_bias_M_2;
	check_error(cuMemAlloc((CUdeviceptr*) &state->weights[0], sizeof(long) * 256*31), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[1], sizeof(long) * 256*2), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[2], sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[3], sizeof(long) * 2), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[4], sizeof(long) * 256 * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[5], sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[6], sizeof(long) * 256 * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[7], sizeof(long) * 256), "cuMemAlloc ", __LINE__);


    //initialize variables
	kbuf_weight_0_T_ent = (long*) kava_alloc(256*31*sizeof(long));
    memcpy(kbuf_weight_0_T_ent, weights[0], 256*31*sizeof(long));
    kbuf_bias_0_ent = (long*) kava_alloc(256*sizeof(long));
    memcpy(kbuf_bias_0_ent, weights[2], 256*sizeof(long));

    kbuf_weight_1_T_ent = (long*) kava_alloc(256*2*sizeof(long));
    memcpy(kbuf_weight_1_T_ent, weights[1], 256*2*sizeof(long));
    kbuf_bias_1_ent = (long*) kava_alloc(2*sizeof(long));
    memcpy(kbuf_bias_1_ent, weights[3], 2*sizeof(long));

    check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[0], kbuf_weight_0_T_ent, sizeof(long) * 256*31), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[1], kbuf_weight_1_T_ent, sizeof(long) * 256*2), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[2], kbuf_bias_0_ent, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[3], kbuf_bias_1_ent, sizeof(long) * 2), "cuMemcpyHtoD", __LINE__);

    kava_free(kbuf_weight_0_T_ent);
    kava_free(kbuf_weight_1_T_ent);
    kava_free(kbuf_bias_0_ent);
    kava_free(kbuf_bias_1_ent);

    //test if +1
    if (weights[4] && weights[5]) {
        //pr_warn("Copying weights for +1\n");
        kbuf_weight_M_1 = (long*) kava_alloc(256*256*sizeof(long));
        memcpy(kbuf_weight_M_1, weights[4], 256*256*sizeof(long));

        kbuf_bias_M_1 = (long*) kava_alloc(256*sizeof(long));
        memcpy(kbuf_bias_M_1, weights[5], 256*sizeof(long));
        
        check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[4], kbuf_weight_M_1, sizeof(long) * 256 * 256), "cuMemcpyHtoD", __LINE__);
        check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[5], kbuf_bias_M_1, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
        kava_free(kbuf_weight_M_1);
        kava_free(kbuf_bias_M_1);
    }
    
    //test if +2
    if (weights[6] && weights[7]) {
        //pr_warn("Copying weights for +2\n");
        kbuf_weight_M_2 = (long*) kava_alloc(256*256*sizeof(long));
        memcpy(kbuf_weight_M_2, weights[6], 256*256*sizeof(long));

        kbuf_bias_M_2 = (long*) kava_alloc(256*sizeof(long));
        memcpy(kbuf_bias_M_2, weights[7], 256*sizeof(long));
        
        check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[6], kbuf_weight_M_2, sizeof(long) * 256 * 256), "cuMemcpyHtoD", __LINE__);
        check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[7], kbuf_bias_M_2, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
        kava_free(kbuf_weight_M_2);
        kava_free(kbuf_bias_M_2);
    }
}


void gpu_cleanup_cuda(struct GPU_weights *state) {
    int i;
    //("Cleaning up dGPU state\n");
    for(i = 0; i <8 ; i++) {
        cuMemFree((CUdeviceptr)state->weights[i]);
    }
    cuMemFree(d_input_vec_i_cuda);
    cuMemFree(d_mid_res_i_cuda);
    cuMemFree(d_mid_res_1_i_cuda);
    cuMemFree(d_mid_res_2_i_cuda);
    cuMemFree(d_final_res_i_cuda);

    if (!inputs_to_gpu) {
        kava_free(inputs_to_gpu);
        inputs_to_gpu = 0;
    }
    if (!gpu_outputs) {
        kava_free(gpu_outputs);
        gpu_outputs = 0;
    }
}

void copy_inputs_to_gpu_cuda(u64 n_inputs) {
    cuMemcpyHtoD(d_input_vec_i_cuda, inputs_to_gpu, sizeof(long) * LEN_INPUT * n_inputs);
}

void copy_results_from_gpu_cuda(u64 n_inputs) {
    cuMemcpyDtoH(gpu_outputs, d_final_res_i_cuda, sizeof(long) * 64 * n_inputs);
}