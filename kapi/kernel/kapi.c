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
#include <linux/types.h>
#include <linux/module.h>
#include <linux/vmalloc.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include "commands.h"
#include "lake_kapi.h"
#include "lake_shm.h"
#include "kargs.h"

// Smart memory allocation: use kmalloc for small memory, vmalloc for large memory
static inline void* fast_alloc(size_t size) {
    // If size is less than one page, use kmalloc (faster)
    if (likely(size < PAGE_SIZE))
        return kmalloc(size, GFP_KERNEL);
    return vmalloc(size);
}

static inline void fast_free(void* ptr, size_t size) {
    if (likely(size < PAGE_SIZE))
        kfree(ptr);
    else
        vfree(ptr);
}

/*
 *
 *   Functions in this file export CUDA symbols.
 *   In general they fill a struct and send it through netlink.
 *   They also choose if they are sync or async calls.
 *   Some have special handling, such as memcpys
 * 
 *   TODO: support netlink copies (not urgent)
 *   TODO: accumulate errors
 */

CUresult CUDAAPI cuInit(unsigned int flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuInit cmd = {
        .API_ID = LAKE_API_cuInit, .flags = flags,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuInit);

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuDeviceGet cmd = {
        .API_ID = LAKE_API_cuDeviceGet, .ordinal = ordinal,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *device = ret.device;
	return ret.res;
}
EXPORT_SYMBOL(cuDeviceGet);

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxCreate cmd = {
        .API_ID = LAKE_API_cuCtxCreate, .flags = flags, .dev = dev
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *pctx = ret.pctx;
	return ret.res;
}
EXPORT_SYMBOL(cuCtxCreate);


CUresult CUDAAPI cuCtxDestroy(CUcontext pctx) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxDestroy cmd = {
        .API_ID = LAKE_API_cuCtxDestroy, .ctx = pctx,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuCtxDestroy);

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleLoad cmd = {
        .API_ID = LAKE_API_cuModuleLoad
    };
    // Use strncpy and ensure null termination
    strncpy(cmd.fname, fname, sizeof(cmd.fname) - 1);
    cmd.fname[sizeof(cmd.fname) - 1] = '\0';
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *module = ret.module;
	return ret.res;
}
EXPORT_SYMBOL(cuModuleLoad);

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleUnload cmd = {
        .API_ID = LAKE_API_cuModuleUnload, .hmod = hmod
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuModuleUnload);

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    struct kernel_args_metadata* meta;
    struct lake_cmd_ret ret;
	struct lake_cmd_cuModuleGetFunction cmd = {
        .API_ID = LAKE_API_cuModuleGetFunction, .hmod = hmod
    };
    // Use strncpy and ensure null termination
    strncpy(cmd.name, name, sizeof(cmd.name) - 1);
    cmd.name[sizeof(cmd.name) - 1] = '\0';
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *hfunc = ret.func;

    //parse and store kargs
    meta = get_kargs(*hfunc);
    kava_parse_function_args(name, meta);

    return ret.res;
}
EXPORT_SYMBOL(cuModuleGetFunction);

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    struct lake_cmd_ret ret;
    struct kernel_args_metadata* meta = get_kargs(f);
    u32 tsize = sizeof(struct lake_cmd_cuLaunchKernel) + meta->total_size;
    // Use smart memory allocation, kmalloc for small memory (faster)
    void* cmd_and_args = fast_alloc(tsize);
    if (unlikely(!cmd_and_args)) {
        ret.res = CUDA_ERROR_OUT_OF_MEMORY;
        return ret.res;
    }
	struct lake_cmd_cuLaunchKernel *cmd = (struct lake_cmd_cuLaunchKernel*) cmd_and_args;
    u8 *args = cmd_and_args + sizeof(struct lake_cmd_cuLaunchKernel);

    cmd->API_ID = LAKE_API_cuLaunchKernel; cmd->f = f; 
    cmd->gridDimX = gridDimX; cmd->gridDimY = gridDimY; cmd->gridDimZ = gridDimZ;
    cmd->blockDimX = blockDimX; cmd->blockDimY = blockDimY; cmd->blockDimZ = blockDimZ;
    cmd->sharedMemBytes = sharedMemBytes; cmd->hStream = hStream; cmd->extra = 0;

    cmd->paramsSize = meta->total_size;
    serialize_args(meta, args, kernelParams);

    lake_send_cmd(cmd_and_args, tsize, CMD_ASYNC, &ret);
    fast_free(cmd_and_args, tsize);
	return ret.res;
}
EXPORT_SYMBOL(cuLaunchKernel);

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemAlloc cmd = {
        .API_ID = LAKE_API_cuMemAlloc, .bytesize = bytesize
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *dptr = ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(cuMemAlloc);

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyHtoD cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoD, .dstDevice = dstDevice, .srcHost = srcHost,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in cuMemcpyHtoD is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyHtoD);

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyDtoH cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoH, .srcDevice = srcDevice,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in cuMemcpyDtoH is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.dstHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyDtoH);

CUresult CUDAAPI cuCtxSynchronize(void) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuCtxSynchronize cmd = {
        .API_ID = LAKE_API_cuCtxSynchronize,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuCtxSynchronize);

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemFree cmd = {
        .API_ID = LAKE_API_cuMemFree, .dptr = dptr
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemFree);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamCreate cmd = {
        .API_ID = LAKE_API_cuStreamCreate, .Flags = Flags
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *phStream = ret.stream;
	return ret.res;
}
EXPORT_SYMBOL(cuStreamCreate);

CUresult CUDAAPI cuStreamDestroy (CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamDestroy cmd = {
        .API_ID = LAKE_API_cuStreamDestroy, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuStreamDestroy);

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuStreamSynchronize cmd = {
        .API_ID = LAKE_API_cuStreamSynchronize, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuStreamSynchronize);

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyHtoDAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoDAsync, .dstDevice = dstDevice, .srcHost = srcHost, 
        .ByteCount = ByteCount, .hStream = hStream
    };
    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in cuMemcpyHtoDAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.srcHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyHtoDAsync);

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemcpyDtoHAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoHAsync, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount, .hStream = hStream
    };
    
    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in cuMemcpyDtoHAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.dstHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(cuMemcpyDtoHAsync);

CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, 
        size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    struct lake_cmd_ret ret;
	struct lake_cmd_cuMemAllocPitch cmd = {
        .API_ID = LAKE_API_cuMemAllocPitch, .WidthInBytes = WidthInBytes,
        .Height = Height, .ElementSizeBytes = ElementSizeBytes
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *dptr = ret.ptr;
    *pPitch = ret.pPitch;
	return ret.res;
}
EXPORT_SYMBOL(cuMemAllocPitch);


/*
 *  Kleio
 */

CUresult CUDAAPI kleioLoadModel(const void *srcHost, size_t len) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioLoadModel cmd = {
        .API_ID = LAKE_API_kleioLoadModel
    };

    // s64 offset = kava_shm_offset(srcHost);
    // if (offset < 0) {
    //     pr_err("srcHost in kleioLoadModel is NOT a kshm pointer (use kava_alloc to fix it)\n");
    //     return CUDA_ERROR_INVALID_VALUE;
    // }
    // cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioLoadModel);

CUresult CUDAAPI kleioInference(const void *srcHost, size_t len, int use_gpu) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioInference cmd = {
        .API_ID = LAKE_API_kleioInference, .len = len,
        .use_gpu = use_gpu
    };
    // s64 offset = kava_shm_offset(srcHost);
    // if (offset < 0) {
    //     pr_err("srcHost in kleioInference is NOT a kshm pointer (use kava_alloc to fix it)\n");
    //     return CUDA_ERROR_INVALID_VALUE;
    // }
    // cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioInference);

CUresult CUDAAPI kleioForceGC(void) {
    struct lake_cmd_ret ret;
	struct lake_cmd_kleioForceGC cmd = {
        .API_ID = LAKE_API_kleioForceGC,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(kleioForceGC);

CUresult CUDAAPI nvmlRunningProcs(int* nproc) {
    struct lake_cmd_ret ret;
	struct lake_cmd_nvmlRunningProcs cmd = {
        .API_ID = LAKE_API_nvmlRunningProcs,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *nproc = (int)ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(nvmlRunningProcs);

CUresult CUDAAPI nvmlUtilRate(int* nproc) {
    struct lake_cmd_ret ret;
	struct lake_cmd_nvmlUtilRate cmd = {
        .API_ID = LAKE_API_nvmlUtilRate,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *nproc = (int)ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(nvmlUtilRate);

hipError_t HIPAPI hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    struct lake_cmd_ret ret;
    struct lake_cmd_hipHostMalloc cmd = {
        .API_ID = LAKE_API_hipHostMalloc,
        .size = size,
        .flags = flags
    };
    pr_err("before send cmd\n");
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    pr_err("after send cmd\n");
    *ptr = ret.ptr;
    return ret.res;
}
EXPORT_SYMBOL(hipHostMalloc);

hipError_t HIPAPI hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
    struct lake_cmd_ret ret;
    struct lake_cmd_hipHostGetDevicePointer cmd = {
        .API_ID = LAKE_API_hipHostGetDevicePointer,
        .flags = flags
    };

      s64 offset = kava_shm_offset(hstPtr);
    if (offset < 0) {
        pr_err("hostPtr in hipHostRegister is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.hstPtr = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *devPtr = ret.ptr;
    return ret.res;
}
EXPORT_SYMBOL(hipHostGetDevicePointer);


hipError_t HIPAPI hipHostFree(void* ptr) {
    struct lake_cmd_ret ret;
    struct lake_cmd_hipHostFree cmd = {
        .API_ID = LAKE_API_hipHostFree, .ptr = ptr
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    return ret.res;
    
}
EXPORT_SYMBOL(hipHostFree);

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags){
    struct lake_cmd_ret ret;
    struct lake_cmd_hipHostRegister cmd = {
        .API_ID = LAKE_API_hipHostRegister,
        .sizeBytes = sizeBytes,
        .flags = flags
    };

    s64 offset = kava_shm_offset(hostPtr);
    if (offset < 0) {
        pr_err("hostPtr in hipHostRegister is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.hostPtr = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    return ret.res;
}
EXPORT_SYMBOL(hipHostRegister);

hipError_t HIPAPI hipHostUnregister(void* hostPtr) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipHostUnregister cmd = {
        .API_ID = LAKE_API_hipHostUnregister
    };
    s64 offset = kava_shm_offset(hostPtr);
    if (offset < 0) {
        pr_err("hostPtr in hipHostUnregister is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.hostPtr = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipHostUnregister);

hipError_t HIPAPI hipInit(unsigned int flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipInit cmd = {
        .API_ID = LAKE_API_hipInit, .flags = flags,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipInit);

hipError_t HIPAPI hipCtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipCtxCreate cmd = {
        .API_ID = LAKE_API_hipCtxCreate, .flags = flags, .dev = dev
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *pctx = ret.pctx;
	return ret.res;
}
EXPORT_SYMBOL(hipCtxCreate);

hipError_t HIPAPI hipModuleGetFunction(hipFunction_t* function, hipModule_t hmod, const char* kname){
    struct kernel_args_metadata* meta;
    struct lake_cmd_ret ret;
	struct lake_cmd_hipModuleGetFunction cmd = {
        .API_ID = LAKE_API_hipModuleGetFunction, .hmod = hmod
    };
    // Use strncpy and ensure null termination
    strncpy(cmd.name, kname, sizeof(cmd.name) - 1);
    cmd.name[sizeof(cmd.name) - 1] = '\0';
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *function = ret.func;

    //parse and store kargs
    meta = get_kargs(*function);
    kava_parse_function_args(kname, meta);

    return ret.res;
}
EXPORT_SYMBOL(hipModuleGetFunction);

hipError_t HIPAPI hipMalloc(void** ptr, size_t size){
    struct lake_cmd_ret ret;
	struct lake_cmd_hipMalloc cmd = {
        .API_ID = LAKE_API_hipMalloc, .bytesize = size
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *ptr = ret.ptr;
	return ret.res;
}
EXPORT_SYMBOL(hipMalloc);

hipError_t HIPAPI hipFree(hipDeviceptr_t dptr) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipFree cmd = {
    .API_ID = LAKE_API_hipFree, .dptr = dptr
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipFree);

    hipError_t HIPAPI hipMemcpyHtoDAsync(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount, hipStream_t hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipMemcpyHtoDAsync cmd = {
        .API_ID = LAKE_API_hipMemcpyHtoDAsync, .dstDevice = dstDevice, .srcHost = srcHost, 
        .ByteCount = ByteCount, .hStream = hStream
    };
    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in HipMemcpyHtoDAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.srcHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipMemcpyHtoDAsync);

hipError_t HIPAPI hipMemcpyHtoD(hipDeviceptr_t dstDevice, const void *srcHost, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipMemcpyHtoD cmd = {
        .API_ID = LAKE_API_hipMemcpyHtoD, .dstDevice = dstDevice, .srcHost = srcHost,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(srcHost);
    if (offset < 0) {
        pr_err("srcHost in HipMemcpyHtoD is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.srcHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipMemcpyHtoD);

hipError_t HIPAPI hipMemcpyDtoH(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipMemcpyDtoH cmd = {
        .API_ID = LAKE_API_hipMemcpyDtoH, .srcDevice = srcDevice,
        .ByteCount = ByteCount
    };

    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in HipMemcpyDtoH is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.dstHost = (void*)offset;
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipMemcpyDtoH);

hipError_t HIPAPI hipDeviceSynchronize(void) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipDeviceSynchronize cmd = {
        .API_ID = LAKE_API_hipDeviceSynchronize,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipDeviceSynchronize);

hipError_t HIPAPI hipModuleLaunchKernel(hipFunction_t f,
    unsigned int gridDimX,
    unsigned int gridDimY,
    unsigned int gridDimZ,
    unsigned int blockDimX,
    unsigned int blockDimY,
    unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    hipStream_t hstream,
    void** kernelParams,
    void** extra){
struct lake_cmd_ret ret;
struct kernel_args_metadata* meta = get_kargs(f);
u32 tsize = sizeof(struct lake_cmd_hipModuleLaunchKernel) + meta->total_size;
// Use smart memory allocation, kmalloc for small memory (faster)
void* cmd_and_args = fast_alloc(tsize);
if (unlikely(!cmd_and_args)) {
    ret.res = hipErrorOutOfMemory;
    return ret.res;
}
struct lake_cmd_hipModuleLaunchKernel *cmd = (struct lake_cmd_hipModuleLaunchKernel*) cmd_and_args;
u8 *args = cmd_and_args + sizeof(struct lake_cmd_hipModuleLaunchKernel);

cmd->API_ID = LAKE_API_hipModuleLaunchKernel; cmd->f = f; 
cmd->gridDimX = gridDimX; cmd->gridDimY = gridDimY; cmd->gridDimZ = gridDimZ;
cmd->blockDimX = blockDimX; cmd->blockDimY = blockDimY; cmd->blockDimZ = blockDimZ;
cmd->sharedMemBytes = sharedMemBytes; cmd->hStream = hstream; cmd->extra = 0;

cmd->paramsSize = meta->total_size;
serialize_args(meta, args, kernelParams);

lake_send_cmd(cmd_and_args, tsize, CMD_ASYNC, &ret);
fast_free(cmd_and_args, tsize);
return ret.res;
}
EXPORT_SYMBOL(hipModuleLaunchKernel);

hipError_t HIPAPI hipDeviceGet(hipdevice_t *device, int ordinal) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipDeviceGet cmd = {
        .API_ID = LAKE_API_hipDeviceGet, .ordinal = ordinal,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *device = ret.device;
	return ret.res;
}
EXPORT_SYMBOL(hipDeviceGet);

hipError_t HIPAPI hipModuleLoad(hipModule_t *module, const char *fname) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipModuleLoad cmd = {
        .API_ID = LAKE_API_hipModuleLoad
    };
    // Use strncpy and ensure null termination
    strncpy(cmd.fname, fname, sizeof(cmd.fname) - 1);
    cmd.fname[sizeof(cmd.fname) - 1] = '\0';
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *module = ret.module;
	return ret.res;
}
EXPORT_SYMBOL(hipModuleLoad);

hipError_t HIPAPI hipStreamCreate(hipStream_t *phStream, unsigned int Flags) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipStreamCreate cmd = {
        .API_ID = LAKE_API_hipStreamCreate, .Flags = Flags
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
    *phStream = ret.stream;
	return ret.res;
}
EXPORT_SYMBOL(hipStreamCreate);

hipError_t HIPAPI hipStreamSynchronize(hipStream_t hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipStreamSynchronize cmd = {
        .API_ID = LAKE_API_hipStreamSynchronize, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipStreamSynchronize);

hipError_t HIPAPI hipStreamDestroy (hipStream_t hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipStreamDestroy cmd = {
        .API_ID = LAKE_API_hipStreamDestroy, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipStreamDestroy);

hipError_t HIPAPI hipCtxDestroy(hipCtx_t pctx) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipCtxDestroy cmd = {
        .API_ID = LAKE_API_hipCtxDestroy, .ctx = pctx,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipCtxDestroy);

hipError_t HIPAPI hipMemcpyDtoHAsync(void *dstHost, hipDeviceptr_t srcDevice, size_t ByteCount, hipStream_t hStream) {
    struct lake_cmd_ret ret;
	struct lake_cmd_hipMemcpyDtoHAsync cmd = {
        .API_ID = LAKE_API_hipMemcpyDtoHAsync, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount, .hStream = hStream
    };
    
    s64 offset = kava_shm_offset(dstHost);
    if (offset < 0) {
        pr_err("dstHost in HipMemcpyDtoHAsync is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return hipErrorInvalidValue;
    }
    cmd.dstHost = (void*)offset;

    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC, &ret);
	return ret.res;
}
EXPORT_SYMBOL(hipMemcpyDtoHAsync);