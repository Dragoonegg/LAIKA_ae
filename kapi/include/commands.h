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
#ifndef __KAPI_COMMANDS_H__
#define __KAPI_COMMANDS_H__

#include "cuda.h"
#include "hip_runtime_api_mini.h"

typedef unsigned int u32;

#define CMD_ASYNC 0
#define CMD_SYNC  1

enum lake_api_ids {
    LAKE_API_cuInit = 0,
    LAKE_API_cuDeviceGet,
    LAKE_API_cuCtxCreate,
    LAKE_API_cuModuleLoad,
    LAKE_API_cuModuleUnload,
    LAKE_API_cuModuleGetFunction = 5,
    LAKE_API_cuLaunchKernel,
    LAKE_API_cuCtxDestroy,
    LAKE_API_cuMemAlloc,
    LAKE_API_cuMemcpyHtoD,
    LAKE_API_cuMemcpyDtoH =10,
    LAKE_API_cuCtxSynchronize,
    LAKE_API_cuMemFree,
    LAKE_API_cuStreamCreate,
    LAKE_API_cuStreamSynchronize,
    LAKE_API_cuStreamDestroy =15,
    LAKE_API_cuMemcpyHtoDAsync,
    LAKE_API_cuMemcpyDtoHAsync,
    LAKE_API_cuMemAllocPitch,
    LAKE_API_kleioLoadModel,
    LAKE_API_kleioInference =20,
    LAKE_API_kleioForceGC,
    LAKE_API_nvmlRunningProcs,
    LAKE_API_nvmlUtilRate,
    LAKE_API_hipInit,
    LAKE_API_hipDeviceGet =25,
    LAKE_API_hipHostMalloc,
    LAKE_API_hipHostGetDevicePointer,
    LAKE_API_hipHostFree,
    LAKE_API_hipHostRegister,
    LAKE_API_hipHostUnregister=30,
    LAKE_API_hipCtxCreate,
    LAKE_API_hipModuleGetFunction,
    LAKE_API_hipMalloc,
    LAKE_API_hipFree,
    LAKE_API_hipMemcpyHtoDAsync=35,
    LAKE_API_hipMemcpyHtoD,
    LAKE_API_hipMemcpyDtoH,
    LAKE_API_hipDeviceSynchronize,
    LAKE_API_hipModuleLaunchKernel,
    LAKE_API_hipModuleLoad=40,
    LAKE_API_hipStreamCreate,
    LAKE_API_hipStreamSynchronize,
    LAKE_API_hipStreamDestroy,
    LAKE_API_hipCtxDestroy,
    LAKE_API_hipMemcpyDtoHAsync
};

struct lake_cmd_ret {
    CUresult res;
    union {
        CUdeviceptr ptr; //u64
        CUdevice device; //int
        CUcontext pctx; //ptr
        CUmodule module; //ptr
        CUfunction func; //ptr
        CUstream stream; //ptr
    };
    size_t pPitch; //malloc pitch ruined everything
};

struct lake_cmd_cuInit {
    u32 API_ID;
    int flags;
};

struct lake_cmd_cuDeviceGet {
    u32 API_ID;
    //CUdevice *device; 
    int ordinal;
};

struct lake_cmd_cuCtxCreate {
    u32 API_ID;
    //CUcontext *pctx; 
    unsigned int flags; 
    CUdevice dev;
};

struct lake_cmd_cuModuleLoad {
    u32 API_ID;
    //CUmodule *module;
    //const char *fname;
    char fname[256];
};

struct lake_cmd_cuModuleUnload {
    u32 API_ID;
    CUmodule hmod;
};

struct lake_cmd_cuModuleGetFunction {
    u32 API_ID;
    //CUfunction *hfunc;
    CUmodule hmod; 
    //char *name;
    char name[256];
};

struct lake_cmd_cuLaunchKernel {
    u32 API_ID;
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    //extra is always null
    void **extra;
    unsigned int paramsSize;
};

struct lake_cmd_cuCtxDestroy {
    u32 API_ID;
    CUcontext ctx;
};

struct lake_cmd_cuMemAlloc {
    u32 API_ID;
    //CUdeviceptr *dptr; 
    size_t bytesize;
};

struct lake_cmd_cuMemcpyHtoD {
    u32 API_ID;
    CUdeviceptr dstDevice; 
    const void *srcHost;
    size_t ByteCount;
};

struct lake_cmd_cuMemcpyDtoH {
    u32 API_ID;
    void *dstHost; 
    CUdeviceptr srcDevice; 
    size_t ByteCount;
};

struct lake_cmd_cuCtxSynchronize {
    u32 API_ID;
};

struct lake_cmd_cuMemFree {
    u32 API_ID;
    CUdeviceptr dptr;
};

struct lake_cmd_cuStreamCreate {
    u32 API_ID;
    //CUstream *phStream;
    unsigned int Flags;
};

struct lake_cmd_cuStreamSynchronize {
    u32 API_ID;
    CUstream hStream;
};

struct lake_cmd_cuStreamDestroy {
    u32 API_ID;
    CUstream hStream;
};

struct lake_cmd_cuMemcpyHtoDAsync {
    u32 API_ID;
    CUdeviceptr dstDevice;
    const void *srcHost; 
    size_t ByteCount; 
    CUstream hStream;
};

struct lake_cmd_cuMemcpyDtoHAsync {
    u32 API_ID;
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
};

struct lake_cmd_cuMemAllocPitch {
    u32 API_ID;
    //CUdeviceptr* dptr;
    //size_t* pPitch;
    size_t WidthInBytes; 
    size_t Height;
    unsigned int ElementSizeBytes;
};

struct lake_cmd_kleioLoadModel {
    u32 API_ID;
    const void *srcHost;
    size_t len;
};

struct lake_cmd_kleioInference {
    u32 API_ID;
    const void *srcHost;
    size_t len;
    int use_gpu;
};

struct lake_cmd_kleioForceGC {
    u32 API_ID;
};

struct lake_cmd_nvmlRunningProcs {
    u32 API_ID;
};

struct lake_cmd_nvmlUtilRate {
    u32 API_ID;
};

struct lake_cmd_hipInit {
    u32 API_ID;
    int flags;
};

struct lake_cmd_hipDeviceGet {
    u32 API_ID;
    //hipDevice_t *device; 
    int ordinal;
};

struct lake_cmd_hipHostMalloc {
    int API_ID;
    size_t size;
    unsigned int flags;
};

struct lake_cmd_hipHostGetDevicePointer {
    int API_ID;
    void* hstPtr;
    unsigned int flags;
};

struct lake_cmd_hipHostFree {
    u32 API_ID;
    void* ptr;
};

struct lake_cmd_hipHostRegister {
    u32 API_ID;
    void* hostPtr;
    size_t sizeBytes;
    unsigned int flags;
};


struct lake_cmd_hipHostUnregister {
    u32 API_ID;
    void* hostPtr;
};

struct lake_cmd_hipCtxCreate {
    u32 API_ID;
    unsigned int flags;
    hipDevice_t dev;
};

struct lake_cmd_hipModuleGetFunction {
    u32 API_ID;
    //CUfunction *hfunc;
    hipModule_t hmod; 
    //char *name;
    char name[256];
};

struct lake_cmd_hipMalloc {
    u32 API_ID;
    size_t bytesize;
};

struct lake_cmd_hipFree {
    u32 API_ID;
    CUdeviceptr dptr;
};

struct lake_cmd_hipMemcpyHtoDAsync {
    u32 API_ID;
    CUdeviceptr dstDevice;
    const void *srcHost; 
    size_t ByteCount; 
    hipStream_t hStream;
};

struct lake_cmd_hipMemcpyHtoD {
    u32 API_ID;
    hipDeviceptr_t dstDevice; 
    const void *srcHost;
    size_t ByteCount;
};

struct lake_cmd_hipMemcpyDtoH {
    u32 API_ID;
    void *dstHost; 
    hipDeviceptr_t srcDevice; 
    size_t ByteCount;
};

struct lake_cmd_hipDeviceSynchronize {
    u32 API_ID;
};

struct lake_cmd_hipModuleLaunchKernel {
    u32 API_ID;
    hipFunction_t f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    hipStream_t hStream;
    //extra is always null
    void **extra;
    unsigned int paramsSize;
};

struct lake_cmd_hipModuleLoad {
    u32 API_ID;
    //CUmodule *module;
    //const char *fname;
    char fname[256];
};

struct lake_cmd_hipStreamCreate {
    u32 API_ID;
    //hipStream_t *phStream;
    unsigned int Flags;
};

struct lake_cmd_hipStreamSynchronize {
    u32 API_ID;
    hipStream_t hStream;
};

struct lake_cmd_hipStreamDestroy {
    u32 API_ID;
    hipStream_t hStream;
};

struct lake_cmd_hipCtxDestroy {
    u32 API_ID;
    hipCtx_t ctx;
};

struct lake_cmd_hipMemcpyDtoHAsync {
    u32 API_ID;
    void *dstHost;
    hipDeviceptr_t srcDevice;
    size_t ByteCount;
    hipStream_t hStream;
};

#endif