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
//#include "cuda.h"
#ifdef __KERNEL__
#include <hip_runtime_api_mini.h>
#include "lake_shm.h"
#else
#include <hip/hip_runtime_api.h>
#endif


#include "variables.h"

hipDeviceptr_t d_input_vec_i;
hipDeviceptr_t d_mid_res_i;
hipDeviceptr_t d_mid_res_1_i;
hipDeviceptr_t d_mid_res_2_i;
hipDeviceptr_t d_final_res_i; 

hipFunction_t batch_linnos_final_layer_kernel = 0;
hipFunction_t batch_linnos_mid_layer_kernel = 0;
hipFunction_t batch_linnos_mid_layer_1_kernel = 0;
hipFunction_t batch_linnos_mid_layer_2_kernel = 0;
hipFunction_t batch_linnos_final_layer_kernel_persistent = 0;
hipFunction_t batch_linnos_mid_layer_kernel_persistent = 0;
hipCtx_t hipctx = 0;

CUdeviceptr d_input_vec_i_cuda;
CUdeviceptr d_mid_res_i_cuda;
CUdeviceptr d_mid_res_1_i_cuda;
CUdeviceptr d_mid_res_2_i_cuda;
CUdeviceptr d_final_res_i_cuda; 

CUfunction batch_linnos_final_layer_kernel_cuda = 0;
CUfunction batch_linnos_mid_layer_kernel_cuda = 0;
CUfunction batch_linnos_mid_layer_1_kernel_cuda = 0;
CUfunction batch_linnos_mid_layer_2_kernel_cuda = 0;
CUcontext cuctx = 0;

long *inputs_to_gpu = 0;
long *gpu_outputs = 0;

//these are host
long *multi_inputs_to_gpu[NUMBER_DEVICES][MAX_DEV_BATCHES];
long *multi_gpu_outputs[NUMBER_DEVICES][MAX_DEV_BATCHES];

hipDeviceptr_t multi_d_input_vec_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
hipDeviceptr_t multi_d_mid_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
hipDeviceptr_t multi_d_mid_res_1_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
hipDeviceptr_t multi_d_mid_res_2_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
hipDeviceptr_t multi_d_final_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];

long *first_weight_ptr_to_dev[NUMBER_DEVICES];

CUstream cu_streams[NUMBER_DEVICES][MAX_DEV_BATCHES];
