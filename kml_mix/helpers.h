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
// CUDA driver
#include "cuda.h"


#define PRINT(...) do { if (1) printk(KERN_INFO __VA_ARGS__); } while (0)
#else
//#include <cuda.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#endif

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


void gpu_init(int dev, hipCtx_t* hipctx);
void gpu_get_cufunc(char* cubin, char* kname, hipFunction_t *func);
void gpu_init_cuda(int dev, CUcontext* cuctx);
void gpu_get_cufunc_cuda(char* cubin, char* kname, CUfunction *func);

#endif