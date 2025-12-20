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
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>
#include "test_weights.h"
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

__global__ void prediction_mid_layer_batch(long *weight_0_T_ent, long *bias_0_ent, long *input_vec_i, long *mid_res_i) { 
	int j, offset;

	int threadId = threadIdx.x;
    int stride = blockDim.x;
	int input_ind = blockIdx.x*LEN_INPUT;
	int blockId = blockIdx.x;
	for (j = threadId, offset=threadId*LEN_INPUT; j < LEN_LAYER_0; j+=stride, offset+=LEN_INPUT*stride) {
		int update_index = blockId*stride + j;
        mid_res_i[update_index] = 0;
		//loop unroll
		mid_res_i[update_index] =  mid_res_i[update_index] + input_vec_i[input_ind + 0] * weight_0_T_ent[offset+0]
		+ input_vec_i[input_ind + 1] * weight_0_T_ent[offset+1]
		+ input_vec_i[input_ind + 2] * weight_0_T_ent[offset+2]
		+ input_vec_i[input_ind + 3] * weight_0_T_ent[offset+3]
		+ input_vec_i[input_ind + 4] * weight_0_T_ent[offset+4]
		+ input_vec_i[input_ind + 5] * weight_0_T_ent[offset+5]
		+ input_vec_i[input_ind + 6] * weight_0_T_ent[offset+6]
		+ input_vec_i[input_ind + 7] * weight_0_T_ent[offset+7]
		+ input_vec_i[input_ind + 8] * weight_0_T_ent[offset+8]
		+ input_vec_i[input_ind + 9] * weight_0_T_ent[offset+9]
		+ input_vec_i[input_ind + 10] * weight_0_T_ent[offset+10]
		+ input_vec_i[input_ind + 11] * weight_0_T_ent[offset+11]
		+ input_vec_i[input_ind + 12] * weight_0_T_ent[offset+12]
		+ input_vec_i[input_ind + 13] * weight_0_T_ent[offset+13]
		+ input_vec_i[input_ind + 14] * weight_0_T_ent[offset+14]
		+ input_vec_i[input_ind + 15] * weight_0_T_ent[offset+15]
		+ input_vec_i[input_ind + 16] * weight_0_T_ent[offset+16]
		+ input_vec_i[input_ind + 17] * weight_0_T_ent[offset+17]
		+ input_vec_i[input_ind+ 18] * weight_0_T_ent[offset+18]
		+ input_vec_i[input_ind + 19] * weight_0_T_ent[offset+19]
		+ input_vec_i[input_ind + 20] * weight_0_T_ent[offset+20]
		+ input_vec_i[input_ind + 21] * weight_0_T_ent[offset+21]
		+ input_vec_i[input_ind + 22] * weight_0_T_ent[offset+22]
		+ input_vec_i[input_ind + 23] * weight_0_T_ent[offset+23]
		+ input_vec_i[input_ind + 24] * weight_0_T_ent[offset+24]
		+ input_vec_i[input_ind + 25] * weight_0_T_ent[offset+25]
		+ input_vec_i[input_ind + 26] * weight_0_T_ent[offset+26]
		+ input_vec_i[input_ind + 27] * weight_0_T_ent[offset+27]
		+ input_vec_i[input_ind + 28] * weight_0_T_ent[offset+28]
		+ input_vec_i[input_ind + 29] * weight_0_T_ent[offset+29]
		+ input_vec_i[input_ind + 30] * weight_0_T_ent[offset+30];

        // apply bias
        mid_res_i[update_index] += bias_0_ent[threadId];
        // relu
        if (mid_res_i[update_index] < 0) {
            mid_res_i[update_index] = 0;
        }		
    }
}

__global__ void prediction_mid_layer_1_batch(long *weight_M_1, long *bias_M_1, long *mid_res_i, long *mid_res_1_i) { 
	int j, offset, k;

	int threadId = threadIdx.x;
    int stride = blockDim.x;
	int input_ind = blockIdx.x*256;
	int blockId = blockIdx.x;
	for (j = threadId, offset=threadId*256; j < LEN_LAYER_0; j+=stride, offset+=256*stride) {
		int update_index = blockId*stride + j;
        mid_res_1_i[update_index] = 0;
		//loop unroll
		for(k = 0; k < 256; k++) {
			mid_res_1_i[update_index] += weight_M_1[offset + k] * mid_res_i[input_ind + k];
		}

        // // apply bias
        mid_res_1_i[update_index] += bias_M_1[threadId];
        // relu
        if (mid_res_1_i[update_index] < 0) {
            mid_res_1_i[update_index] = 0;
        }		
    }
}

__global__ void prediction_mid_layer_2_batch(long *weight_M_2, long *bias_M_2, long *mid_res_1_i, long *mid_res_2_i) { 
	int j, offset, k;

	int threadId = threadIdx.x;
    int stride = blockDim.x;
	int input_ind = blockIdx.x*256;
	int blockId = blockIdx.x;
	for (j = threadId, offset=threadId*256; j < LEN_LAYER_0; j+=stride, offset+=256*stride) {
		int update_index = blockId*stride + j;
        mid_res_2_i[update_index] = 0;
		//loop unroll
		for(k = 0; k < 256; k++) {
			mid_res_2_i[update_index] += weight_M_2[offset + k] * mid_res_1_i[input_ind + k];
		}

        // apply bias
        mid_res_2_i[update_index] += bias_M_2[threadId];
        // relu
        if (mid_res_2_i[update_index] < 0) {
            mid_res_2_i[update_index] = 0;
        }		
    }
}

__global__ void prediction_final_layer_batch(long *weight_1_T_ent, long *bias_1_ent, long *mid_res_i, long *dd_final_res_i) {
	int index = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
	int k;
	int update_index = index*dim + threadId;
	if (threadId < 32) {
		dd_final_res_i[update_index] = 0;
		for(k = threadId; k<LEN_LAYER_0; k = k + 32) {
			dd_final_res_i[update_index] =  dd_final_res_i[update_index] + mid_res_i[index*LEN_LAYER_0 + k] * weight_1_T_ent[k];
		}
	} else {
		dd_final_res_i[update_index] = 0;
		for(k = threadId - 32; k<LEN_LAYER_0; k = k + 32) {
			dd_final_res_i[update_index] =  dd_final_res_i[update_index] + mid_res_i[index*LEN_LAYER_0 + k] * weight_1_T_ent[k+256];
		}
	}
	__syncthreads();
	if (threadId == 0) {
		update_index = index*dim;
		for(int i = 1; i < 32; i++) {
			dd_final_res_i[update_index] = dd_final_res_i[update_index] + dd_final_res_i[update_index + i];
		}
		dd_final_res_i[update_index] =  dd_final_res_i[update_index] + bias_1_ent[0];
	}
	if(threadId == 32) {
		update_index = index*dim + 32;
		for(int i = 1; i < 32; i++) {
			dd_final_res_i[update_index] = dd_final_res_i[update_index] + dd_final_res_i[update_index + i];
		} 
		dd_final_res_i[update_index] =  dd_final_res_i[update_index] + bias_1_ent[1];
	}
}

__global__ void prediction_mid_layer_batch_persistent(long *weight_0_T_ent, long *bias_0_ent, long *input_vec_i, long *mid_res_i, int *task_flag, int *quit_flag) { 
	int j, offset;

	int threadId = threadIdx.x;
    int stride = blockDim.x;
	int input_ind = blockIdx.x*LEN_INPUT;
	int blockId = blockIdx.x;
	
	// Debug info: only the first thread prints startup information
	// if (threadId == 0 && blockIdx.x == 0) {
	// 	printf("[DEBUG] prediction_mid_layer_batch kernel started successfully!\n");
	// }
	while (true)
	{	__threadfence_system();
		if (*quit_flag) {break;	
		}
		if(*task_flag == 1) {
			for (j = threadId, offset=threadId*LEN_INPUT; j < LEN_LAYER_0; j+=stride, offset+=LEN_INPUT*stride) {
				int update_index = blockId*stride + j;
				mid_res_i[update_index] = 0;
				//loop unroll
				mid_res_i[update_index] =  mid_res_i[update_index] + input_vec_i[input_ind + 0] * weight_0_T_ent[offset+0]
				+ input_vec_i[input_ind + 1] * weight_0_T_ent[offset+1]
				+ input_vec_i[input_ind + 2] * weight_0_T_ent[offset+2]
				+ input_vec_i[input_ind + 3] * weight_0_T_ent[offset+3]
				+ input_vec_i[input_ind + 4] * weight_0_T_ent[offset+4]
				+ input_vec_i[input_ind + 5] * weight_0_T_ent[offset+5]
				+ input_vec_i[input_ind + 6] * weight_0_T_ent[offset+6]
				+ input_vec_i[input_ind + 7] * weight_0_T_ent[offset+7]
				+ input_vec_i[input_ind + 8] * weight_0_T_ent[offset+8]
				+ input_vec_i[input_ind + 9] * weight_0_T_ent[offset+9]
				+ input_vec_i[input_ind + 10] * weight_0_T_ent[offset+10]
				+ input_vec_i[input_ind + 11] * weight_0_T_ent[offset+11]
				+ input_vec_i[input_ind + 12] * weight_0_T_ent[offset+12]
				+ input_vec_i[input_ind + 13] * weight_0_T_ent[offset+13]
				+ input_vec_i[input_ind + 14] * weight_0_T_ent[offset+14]
				+ input_vec_i[input_ind + 15] * weight_0_T_ent[offset+15]
				+ input_vec_i[input_ind + 16] * weight_0_T_ent[offset+16]
				+ input_vec_i[input_ind + 17] * weight_0_T_ent[offset+17]
				+ input_vec_i[input_ind+ 18] * weight_0_T_ent[offset+18]
				+ input_vec_i[input_ind + 19] * weight_0_T_ent[offset+19]
				+ input_vec_i[input_ind + 20] * weight_0_T_ent[offset+20]
				+ input_vec_i[input_ind + 21] * weight_0_T_ent[offset+21]
				+ input_vec_i[input_ind + 22] * weight_0_T_ent[offset+22]
				+ input_vec_i[input_ind + 23] * weight_0_T_ent[offset+23]
				+ input_vec_i[input_ind + 24] * weight_0_T_ent[offset+24]
				+ input_vec_i[input_ind + 25] * weight_0_T_ent[offset+25]
				+ input_vec_i[input_ind + 26] * weight_0_T_ent[offset+26]
				+ input_vec_i[input_ind + 27] * weight_0_T_ent[offset+27]
				+ input_vec_i[input_ind + 28] * weight_0_T_ent[offset+28]
				+ input_vec_i[input_ind + 29] * weight_0_T_ent[offset+29]
				+ input_vec_i[input_ind + 30] * weight_0_T_ent[offset+30];

				// apply bias
				mid_res_i[update_index] += bias_0_ent[threadId];
				// relu
				if (mid_res_i[update_index] < 0) {
					mid_res_i[update_index] = 0;
				}		
    		}
			// Set completion flag - only the first thread sets it
			if (threadId == 0 && blockIdx.x == 0) {
				*task_flag = 2; // Set to 2 to indicate task completion
				__threadfence_system(); // Ensure visible to host
			}
		}
		else{
			//__builtin_amdgcn_s_sleep(10000);        
		}
	}
}



__global__ void prediction_final_layer_batch_persistent(
    long *weight_1_T_ent, long *bias_1_ent, long *mid_res_i, long *dd_final_res_i,
    int *task_flag, int *quit_flag)
{
    #define THREADS_PER_BLOCK 64
    #define WARP_SIZE 32
    int index = blockIdx.x;
    int threadId = threadIdx.x;
    int dim = blockDim.x;
    int k;

    // Only allow 64 threads
    if (dim != THREADS_PER_BLOCK) {printf("dim != THREADS_PER_BLOCK\n"); return;}

    while (true) {
        __threadfence_system();
        if (*quit_flag) break;
        if (*task_flag == 2) {
			//+0 task_flag =2
			//+1 task_flag =3
			//+2 task_flag =4
            // Each thread initializes its own result
            long sum = 0;
            if (threadId < WARP_SIZE) {
                for (k = threadId; k < LEN_LAYER_0; k += WARP_SIZE) {
                    sum += mid_res_i[index * LEN_LAYER_0 + k] * weight_1_T_ent[k];
                }
            } else if (threadId < 2 * WARP_SIZE) {
                for (k = threadId - WARP_SIZE; k < LEN_LAYER_0; k += WARP_SIZE) {
                    sum += mid_res_i[index * LEN_LAYER_0 + k] * weight_1_T_ent[k + 256];
                }
            }
            // Write temporary result
            dd_final_res_i[index * dim + threadId] = sum;

            __syncthreads();
            // Reduce first 32 threads
            if (threadId == 0) {
                long total = 0;
                for (int i = 0; i < WARP_SIZE; i++) {
                    total += dd_final_res_i[index * dim + i];
                }
                dd_final_res_i[index * dim] = total + bias_1_ent[0];
            }
            // Reduce last 32 threads
            if (threadId == WARP_SIZE) {
                long total = 0;
                for (int i = 0; i < WARP_SIZE; i++) {
                    total += dd_final_res_i[index * dim + WARP_SIZE + i];
                }
                dd_final_res_i[index * dim + WARP_SIZE] = total + bias_1_ent[1];
            }

            

            if (threadId == 0 && blockIdx.x == 0) {
                *task_flag = 8; // Indicate completion
                __threadfence_system();
            }
        }
    }
}
