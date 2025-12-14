#include <cuda.h>
#include <hip_runtime_api_mini.h>
#include <inttypes.h>
#include <stdio.h>
#include <nvml.h>
#include "commands.h"
#include "lake_shm.h"
#include "lake_kapi.h"
#include "kargs.h"
#include "handler_helpers.h"

#define DRY_RUN 0

/*********************
 *  cuInit    
 *********************/
static int lake_handler_cuInit(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuInit *cmd = (struct lake_cmd_cuInit *) buf;
    cmd_ret->res = cuInit(cmd->flags);
    return 0;
}

/*********************
 *  cuDeviceGet   
 *********************/
static int lake_handler_cuDeviceGet(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuDeviceGet *cmd = (struct lake_cmd_cuDeviceGet *) buf;
    cmd_ret->res = cuDeviceGet(&cmd_ret->device, cmd->ordinal);
    return 0;
}

/*********************
 *  cuCtxCreate   
 *********************/
static int lake_handler_cuCtxCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuCtxCreate *cmd = (struct lake_cmd_cuCtxCreate *) buf;
    cmd_ret->res = cuCtxCreate_v2(&cmd_ret->pctx, cmd->flags, cmd->dev);
    return 0;
}

/*********************
 *  cuModuleLoad   
 *********************/
static int lake_handler_cuModuleLoad(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuModuleLoad *cmd = (struct lake_cmd_cuModuleLoad *) buf;
    cmd_ret->res = cuModuleLoad(&cmd_ret->module, cmd->fname);
    return 0;
}

/*********************
 *  cuModuleUnload   
 *********************/
static int lake_handler_cuModuleUnload(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuModuleUnload *cmd = (struct lake_cmd_cuModuleUnload *) buf;
    cmd_ret->res = cuModuleUnload(cmd->hmod);
    return 0;
}

/*********************
 *  cuModuleGetFunction   
 *********************/
static int lake_handler_cuModuleGetFunction(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuModuleGetFunction *cmd = (struct lake_cmd_cuModuleGetFunction *) buf;
    cmd_ret->res = cuModuleGetFunction(&cmd_ret->func, cmd->hmod, cmd->name);
    struct kernel_args_metadata* meta = get_kargs(cmd_ret->func);
    kava_parse_function_args(cmd->name, meta);
    return 0;
}

/*********************
 *  cuLaunchKernel   
 *********************/
static int lake_handler_cuLaunchKernel(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuLaunchKernel *cmd = (struct lake_cmd_cuLaunchKernel *) buf;
    struct kernel_args_metadata* meta = get_kargs(cmd->f);
    uint8_t *serialized = ((u8*)buf) + sizeof(struct lake_cmd_cuLaunchKernel);
    void* args = malloc(meta->func_argc * sizeof(void*));
    construct_args(meta, args, serialized), 
    cmd_ret->res = cuLaunchKernel(cmd->f, cmd->gridDimX, cmd->gridDimY,
        cmd->gridDimZ, cmd->blockDimX, cmd->blockDimY, cmd->blockDimZ, cmd->sharedMemBytes,
        cmd->hStream, args, cmd->extra);
    
    return 0;
}

/*********************
 *  cuCtxDestroy   
 *********************/
static int lake_handler_cuCtxDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuCtxDestroy *cmd = (struct lake_cmd_cuCtxDestroy *) buf;
    cmd_ret->res = cuCtxDestroy(cmd->ctx);
    return 0;
}

/*********************
 *  cuMemAlloc   
 *********************/
static int lake_handler_cuMemAlloc(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemAlloc *cmd = (struct lake_cmd_cuMemAlloc *) buf;
    cmd_ret->res = cuMemAlloc(&cmd_ret->ptr, cmd->bytesize);
    return 0;
}

/*********************
 *  cuMemcpyHtoD   
 *********************/
static int lake_handler_cuMemcpyHtoD(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemcpyHtoD *cmd = (struct lake_cmd_cuMemcpyHtoD *) buf;
    cmd_ret->res = cuMemcpyHtoD(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount);
    return 0;
}

/*********************
 *  cuMemcpyDtoH   
 *********************/
static int lake_handler_cuMemcpyDtoH(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemcpyDtoH *cmd = (struct lake_cmd_cuMemcpyDtoH *) buf;
    cmd_ret->res = cuMemcpyDtoH(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount);
    return 0;
}

/*********************
 *  cuCtxSynchronize   
 *********************/
static int lake_handler_cuCtxSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuCtxSynchronize *cmd = (struct lake_cmd_cuCtxSynchronize *) buf;
    cmd_ret->res = cuCtxSynchronize();
    return 0;
}

/*********************
 *  cuMemFree   
 *********************/
static int lake_handler_cuMemFree(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemFree *cmd = (struct lake_cmd_cuMemFree *) buf;
    cmd_ret->res = cuMemFree(cmd->dptr);
    return 0;
}

/*********************
 *  cuStreamCreate   
 *********************/
static int lake_handler_cuStreamCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuStreamCreate *cmd = (struct lake_cmd_cuStreamCreate *) buf;
    cmd_ret->res = cuStreamCreate(&cmd_ret->stream, cmd->Flags);
    return 0;
}

/*********************
 *  cuStreamSynchronize   
 *********************/
static int lake_handler_cuStreamSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuStreamSynchronize *cmd = (struct lake_cmd_cuStreamSynchronize *) buf;
    cmd_ret->res = cuStreamSynchronize(cmd->hStream);
    return 0;
}

/*********************
 *  cuStreamDestroy   
 *********************/
static int lake_handler_cuStreamDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuStreamDestroy *cmd = (struct lake_cmd_cuStreamDestroy *) buf;
    cmd_ret->res = cuStreamDestroy(cmd->hStream);
    return 0;
}

/*********************
 *  cuMemcpyHtoDAsync   
 *********************/
static int lake_handler_cuMemcpyHtoDAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemcpyHtoDAsync *cmd = (struct lake_cmd_cuMemcpyHtoDAsync *) buf;
    cmd_ret->res = cuMemcpyHtoDAsync(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount, cmd->hStream);
    return 0;
}

/*********************
 *  cuMemcpyDtoHAsync   
 *********************/
static int lake_handler_cuMemcpyDtoHAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemcpyDtoHAsync *cmd = (struct lake_cmd_cuMemcpyDtoHAsync *) buf;
    cmd_ret->res = cuMemcpyDtoHAsync(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount, cmd->hStream);
    return 0;
}

/*********************
 *  cuMemAllocPitch   
 *********************/
static int lake_handler_cuMemAllocPitch(void* buf, struct lake_cmd_ret* cmd_ret) {
        struct lake_cmd_cuMemAllocPitch *cmd = (struct lake_cmd_cuMemAllocPitch *) buf;
    cmd_ret->res = cuMemAllocPitch(&cmd_ret->ptr, &cmd_ret->pPitch, cmd->WidthInBytes,
        cmd->Height, cmd->ElementSizeBytes);
    return 0;
}

/*********************
 *  kleioLoadModel   
 *********************/
//int kleio_load_model(const char *filepath)
static int lake_handler_kleioLoadModel(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_kleioLoadModel *cmd = (struct lake_cmd_kleioLoadModel *) buf;
    // Python support removed - kleio_load_model(0);
    cmd_ret->res = 0;
    return 0;
}

static int lake_handler_kleioInference(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_kleioInference *cmd = (struct lake_cmd_kleioInference *) buf;
    // Python support removed - kleio_inference(0, cmd->len, cmd->use_gpu);
    cmd_ret->res = 0;
    return 0;
}

static int lake_handler_kleioForceGC(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_kleioForceGC *cmd = (struct lake_cmd_kleioForceGC *) buf;
    cmd_ret->res = 0;
    return 0;
}

static int lake_handler_nvmlRunningProcs(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_nvmlRunningProcs *cmd = (struct lake_cmd_nvmlRunningProcs *) buf;
    cmd_ret->ptr = (int) nvml_get_procs_running();
    cmd_ret->res = 0;
    return 0;
}

static int lake_handler_nvmlUtilRate(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_nvmlUtilRate *cmd = (struct lake_cmd_nvmlUtilRate *) buf;
    cmd_ret->ptr = (int) nvml_get_util_rate();
    cmd_ret->res = 0;
    return 0;
}


/*********************
 *  hipInit    
 *********************/
static int lake_handler_hipInit(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipInit *cmd = (struct lake_cmd_hipInit *) buf;
    cmd_ret->res = hipInit(cmd->flags);
    return 0;
}

/*********************
*  hipDeviceGet   
*********************/
static int lake_handler_hipDeviceGet(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipDeviceGet *cmd = (struct lake_cmd_hipDeviceGet *) buf;
cmd_ret->res = hipDeviceGet(&cmd_ret->device, cmd->ordinal);
return 0;
}

/*********************
*  hipHostMalloc   
*********************/
static int lake_handler_hipHostMalloc(void* buf, struct lake_cmd_ret* cmd_ret) {
struct lake_cmd_hipHostMalloc *cmd = (struct lake_cmd_hipHostMalloc *) buf;
void* ptr = NULL;
cmd_ret->res = hipHostMalloc(&ptr, cmd->size, cmd->flags);
//cmd_ret->ptr = (CUdeviceptr)ptr;
return 0;
}

/*********************
*  hipHostGetDevicePointer   
*********************/
static int lake_handler_hipHostGetDevicePointer(void* buf, struct lake_cmd_ret* cmd_ret) {
struct lake_cmd_hipHostGetDevicePointer *cmd = (struct lake_cmd_hipHostGetDevicePointer *) buf;
void* devPtr = NULL;
cmd_ret->res = hipHostGetDevicePointer(&devPtr, lake_shm_address(cmd->hstPtr), cmd->flags);
cmd_ret->ptr = (CUdeviceptr)devPtr;
return 0;
}

/*********************
*  hipHostFree   
*********************/
static int lake_handler_hipHostFree(void* buf, struct lake_cmd_ret* cmd_ret) {
struct lake_cmd_hipHostFree *cmd = (struct lake_cmd_hipHostFree *) buf;
cmd_ret->res = hipHostFree(cmd->ptr);
return 0;
}

static int lake_handler_hipHostRegister(void* buf, struct lake_cmd_ret* cmd_ret) {
struct lake_cmd_hipHostRegister *cmd = (struct lake_cmd_hipHostRegister *) buf;
cmd_ret->res = hipHostRegister(lake_shm_address(cmd->hostPtr), cmd->sizeBytes, cmd->flags);
return 0;
}

static int lake_handler_hipHostUnregister(void* buf, struct lake_cmd_ret* cmd_ret) {
struct lake_cmd_hipHostUnregister *cmd = (struct lake_cmd_hipHostUnregister *) buf;
cmd_ret->res = hipHostUnregister(lake_shm_address(cmd->hostPtr));
return 0;
}

static int lake_handler_hipCtxCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipCtxCreate *cmd = (struct lake_cmd_hipCtxCreate *) buf;
    cmd_ret->res = hipCtxCreate(&cmd_ret->pctx, cmd->flags, cmd->dev);
    return 0;
}

static int lake_handler_hipModuleGetFunction(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipModuleGetFunction *cmd = (struct lake_cmd_hipModuleGetFunction *) buf;
    cmd_ret->res = hipModuleGetFunction(&cmd_ret->func, cmd->hmod, cmd->name);
    struct kernel_args_metadata* meta = get_kargs(cmd_ret->func);
    kava_parse_function_args(cmd->name, meta);
    return 0;
}

static int lake_handler_hipMalloc(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipMalloc *cmd = (struct lake_cmd_hipMalloc *) buf;
    cmd_ret->res = hipMalloc(&cmd_ret->ptr, cmd->bytesize);
    return 0;
}

static int lake_handler_hipFree(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipFree *cmd = (struct lake_cmd_hipFree *) buf;
    cmd_ret->res = hipFree(cmd->dptr);
    return 0;
}

static int lake_handler_hipMemcpyHtoDAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipMemcpyHtoDAsync *cmd = (struct lake_cmd_hipMemcpyHtoDAsync *) buf;
cmd_ret->res = hipMemcpyHtoDAsync(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount, cmd->hStream);
return 0;
}
static int lake_handler_hipMemcpyHtoD(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipMemcpyHtoD *cmd = (struct lake_cmd_hipMemcpyHtoD *) buf;
cmd_ret->res = hipMemcpyHtoD(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount);
return 0;
}
static int lake_handler_hipMemcpyDtoH(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipMemcpyDtoH *cmd = (struct lake_cmd_hipMemcpyDtoH *) buf;
cmd_ret->res = hipMemcpyDtoH(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount);
return 0;
}
static int lake_handler_hipDeviceSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipDeviceSynchronize *cmd = (struct lake_cmd_hipDeviceSynchronize *) buf;
cmd_ret->res = hipDeviceSynchronize();
return 0;
}

static int lake_handler_hipModuleLaunchKernel(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipModuleLaunchKernel *cmd = (struct lake_cmd_hipModuleLaunchKernel *) buf;
struct kernel_args_metadata* meta = get_kargs(cmd->f);
uint8_t *serialized = ((u8*)buf) + sizeof(struct lake_cmd_hipModuleLaunchKernel);
void* args = malloc(meta->func_argc * sizeof(void*));
construct_args(meta, args, serialized), 
cmd_ret->res = hipModuleLaunchKernel(cmd->f, cmd->gridDimX, cmd->gridDimY,
    cmd->gridDimZ, cmd->blockDimX, cmd->blockDimY, cmd->blockDimZ, cmd->sharedMemBytes,
    cmd->hStream, args, cmd->extra);

return 0;
}

static int lake_handler_hipModuleLoad(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipModuleLoad *cmd = (struct lake_cmd_hipModuleLoad *) buf;
cmd_ret->res = hipModuleLoad(&cmd_ret->module, cmd->fname);
return 0;
}

static int lake_handler_hipStreamCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipStreamCreate *cmd = (struct lake_cmd_hipStreamCreate *) buf;
cmd_ret->res = hipStreamCreate(&cmd_ret->stream, cmd->Flags);
return 0;
}

static int lake_handler_hipStreamSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipStreamSynchronize *cmd = (struct lake_cmd_hipStreamSynchronize *) buf;
cmd_ret->res = hipStreamSynchronize(cmd->hStream);
return 0;
}

static int lake_handler_hipStreamDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipStreamDestroy *cmd = (struct lake_cmd_hipStreamDestroy *) buf;
cmd_ret->res = hipStreamDestroy(cmd->hStream);
return 0;
}

static int lake_handler_hipCtxDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipCtxDestroy *cmd = (struct lake_cmd_hipCtxDestroy *) buf;
cmd_ret->res = hipCtxDestroy(cmd->ctx);
return 0;
}

static int lake_handler_hipMemcpyDtoHAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_hipMemcpyDtoHAsync *cmd = (struct lake_cmd_hipMemcpyDtoHAsync *) buf;
cmd_ret->res = hipMemcpyDtoHAsync(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount, cmd->hStream);
return 0;
}

/*********************
 * 
 *  END OF HANDLERS
 *    
 *********************/

//order matters, need to match enum in src/kapi/include/commands.h
static int (*kapi_handlers[])(void* buf, struct lake_cmd_ret* cmd_ret) = {
    lake_handler_cuInit,
    lake_handler_cuDeviceGet,
    lake_handler_cuCtxCreate,
    lake_handler_cuModuleLoad,
    lake_handler_cuModuleUnload,
    lake_handler_cuModuleGetFunction,
    lake_handler_cuLaunchKernel,
    lake_handler_cuCtxDestroy,
    lake_handler_cuMemAlloc,
    lake_handler_cuMemcpyHtoD,
    lake_handler_cuMemcpyDtoH,
    lake_handler_cuCtxSynchronize,
    lake_handler_cuMemFree,
    lake_handler_cuStreamCreate,
    lake_handler_cuStreamSynchronize,
    lake_handler_cuStreamDestroy,
    lake_handler_cuMemcpyHtoDAsync,
    lake_handler_cuMemcpyDtoHAsync,
    lake_handler_cuMemAllocPitch,
    lake_handler_kleioLoadModel,
    lake_handler_kleioInference,
    lake_handler_kleioForceGC,
    lake_handler_nvmlRunningProcs,
    lake_handler_nvmlUtilRate,
    lake_handler_hipInit,
    lake_handler_hipDeviceGet,
    lake_handler_hipHostMalloc,
    lake_handler_hipHostGetDevicePointer,
    lake_handler_hipHostFree,
    lake_handler_hipHostRegister,
    lake_handler_hipHostUnregister,
    lake_handler_hipCtxCreate,
    lake_handler_hipModuleGetFunction,
    lake_handler_hipMalloc,
    lake_handler_hipFree,
    lake_handler_hipMemcpyHtoDAsync,
    lake_handler_hipMemcpyHtoD,
    lake_handler_hipMemcpyDtoH,
    lake_handler_hipDeviceSynchronize,
    lake_handler_hipModuleLaunchKernel,
    lake_handler_hipModuleLoad,
    lake_handler_hipStreamCreate,
    lake_handler_hipStreamSynchronize,
    lake_handler_hipStreamDestroy,
    lake_handler_hipCtxDestroy,
    lake_handler_hipMemcpyDtoHAsync
};

void lake_handle_cmd(void* buf, struct lake_cmd_ret* cmd_ret) {
    uint32_t cmd_id = *((uint32_t*) buf);
    kapi_handlers[cmd_id](buf, cmd_ret);
    if(cmd_ret->res != 0) {
        printf("Command %u returned error %d\n", cmd_id, cmd_ret->res);
    }
}

