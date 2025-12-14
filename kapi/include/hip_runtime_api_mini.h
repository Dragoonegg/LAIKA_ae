#ifndef __KAVA_HIP_RUNTIME_API_H__
#define __KAVA_HIP_RUNTIME_API_H__

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <sys/types.h>
#endif

#ifdef _WIN32
#define HIPAPI __stdcall
#else
#define HIPAPI
#endif


#ifdef __cplusplus
extern "C" {
#endif

// 错误码
// Ignoring error-code return values from hip APIs is discouraged. On C++17,
// we can make that yield a warning
#if __cplusplus >= 201703L
#define __HIP_NODISCARD [[nodiscard]]
#else
#define __HIP_NODISCARD
#endif

#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long hipdeviceptr_t;
#else
typedef unsigned int hipdeviceptr_t;
#endif
typedef int hipdevice_t; //int



/**
 * HIP error type
 *
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in
// NVCC and HCC paths Also update the hipCUDAErrorTohipError function in NVCC path.

typedef enum __HIP_NODISCARD hipError_t {
    hipSuccess = 0,  ///< Successful completion.
    hipErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,   ///< out of memory range.
    // Deprecated
    hipErrorMemoryAllocation = 2,  ///< Memory allocation error.
    hipErrorNotInitialized = 3,    ///< Invalid not initialized
    // Deprecated
    hipErrorInitializationError = 3,
    hipErrorDeinitialized = 4,      ///< Deinitialized
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,  ///< Invalide configuration
    hipErrorInvalidPitchValue = 12,   ///< Invalid pitch value
    hipErrorInvalidSymbol = 13,   ///< Invalid symbol
    hipErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,  ///< Invalid device function
    hipErrorNoDevice = 100,  ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,  ///< DeviceID must be in range from 0 to compute-devices.
    hipErrorInvalidImage = 200,   ///< Invalid image
    hipErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    // Deprecated
    hipErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,   ///< Unsupported limit
    hipErrorContextAlreadyInUse = 216,   ///< The context is already in use
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,   ///< Invalid source.
    hipErrorFileNotFound = 301,   ///< the file is not found.
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,   ///< Failed to initialize shared object.
    hipErrorOperatingSystem = 304,   ///< Not the correct operating system
    hipErrorInvalidHandle = 400,  ///< Invalide handle
    // Deprecated
    hipErrorInvalidResourceHandle = 400,  ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,   ///< Not found
    hipErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from hipSuccess (which indicates completion).  APIs that return
                             ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,   ///< Timeout for the launch.
    hipErrorPeerAccessAlreadyEnabled = 704,  ///< Peer access was already enabled from the current
                                             ///< device.
    hipErrorPeerAccessNotEnabled = 705,  ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,   ///< The process is active.
    hipErrorContextIsDestroyed = 709,   ///< The context is already destroyed
    hipErrorAssert = 710,  ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,  ///< Produced when trying to lock a page-locked
                                                ///< memory.
    hipErrorHostMemoryNotRegistered = 713,  ///< Produced when trying to unlock a non-page-locked
                                            ///< memory.
    hipErrorLaunchFailure = 719,  ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge = 720,  ///< This error indicates that the number of blocks
                                              ///< launched per grid for a kernel that was launched
                                              ///< via cooperative launch APIs exceeds the maximum
                                              ///< number of allowed blocks for the current device.
    hipErrorNotSupported = 801,  ///< Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies  are allowed
                                           ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the hipStreamCaptureModeRelaxed argument to
                                             ///< hipStreamBeginCapture was passed to
                                             ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraintsspecific to instantiated graph
                                           ///< update.
    hipErrorUnknown = 999,  ///< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    hipErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    hipErrorTbd  ///< Marker that more error codes are needed.
} hipError_t;

#undef __HIP_NODISCARD

typedef struct CUctx_st *CUcontext;                       /**< CUDA context */
typedef struct CUmod_st *CUmodule;                        /**< CUDA module */
typedef struct CUfunc_st *CUfunction; 
typedef struct CUstream_st *CUstream;  

  typedef CUcontext hipCtx_t;
  typedef CUmodule hipModule_t;
  typedef CUfunction hipFunction_t;
  typedef CUstream hipStream_t;

typedef int hipDevice_t;
//typedef struct ihipCtx_t* hipCtx_t;
//typedef struct ihipModule_t* hipModule_t;
//typedef struct ihipModuleSymbol_t* hipFunction_t;
//typedef struct ihipStream_t* hipStream_t;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long hipDeviceptr_t;


extern hipError_t HIPAPI hipCtxCreate(hipCtx_t *pctx, unsigned int flags, hipDevice_t dev);
extern hipError_t HIPAPI hipModuleLoad(hipModule_t* module, const char* fname);
extern hipError_t HIPAPI hipModuleGetFunction(hipFunction_t* function, hipModule_t hmod, const char* kname);
extern hipError_t HIPAPI hipMalloc(void** ptr, size_t size);
extern hipError_t HIPAPI hipMemcpyHtoD(hipDeviceptr_t dst, const void* src, size_t sizeBytes);
extern hipError_t HIPAPI hipModuleLaunchKernel(hipFunction_t f,
                                 unsigned int gridDimX,
                                 unsigned int gridDimY,
                                 unsigned int gridDimZ,
                                 unsigned int blockDimX,
                                 unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 hipStream_t hstream,
                                 void** kernelParams,
                                 void** extra);
extern hipError_t HIPAPI hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t sizeBytes);
extern hipError_t HIPAPI hipInit(unsigned int flags);
extern hipError_t HIPAPI hipDeviceGet(hipdevice_t *device, int ordinal) ;
extern hipError_t HIPAPI hipDeviceSynchronize(void);
extern hipError_t HIPAPI hipFree(hipDeviceptr_t dptr);

extern hipError_t HIPAPI hipModuleUnload(CUmodule hmod);
extern hipError_t HIPAPI hipCtxDestroy(hipCtx_t pctx);
extern hipError_t HIPAPI hipStreamCreate(CUstream *phStream, unsigned int Flags);
extern hipError_t HIPAPI hipStreamDestroy(CUstream hStream);
extern hipError_t HIPAPI hipStreamSynchronize(CUstream hStream);
extern hipError_t HIPAPI hipMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
extern hipError_t HIPAPI hipMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
extern hipError_t HIPAPI hipMemAllocPitch(CUdeviceptr* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes);
//zero copy
extern hipError_t HIPAPI hipHostMalloc(void** ptr, size_t size, unsigned int flags);
extern hipError_t HIPAPI hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);
extern hipError_t HIPAPI hipHostFree(void* ptr);
extern hipError_t HIPAPI hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
extern hipError_t HIPAPI hipHostUnregister(void* hostPtr);
#ifdef __cplusplus
}
#endif

#endif // __KAVA_HIP_RUNTIME_API_H__


#define hipHostMallocMapped  0x2
#define hipHostAllocMapped  0x2

#define hipHostMallocWriteCombined 0x4
#define hipHostAllocWriteCombined 0x4

//Flags that can be used with hipHostRegister.
/** Memory is Mapped and Portable.*/
#define hipHostRegisterDefault 0x0

/** Memory is considered registered by all contexts.*/
#define hipHostRegisterPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #hipHostGetDevicePointer.*/
#define hipHostRegisterMapped  0x2

/** Not supported.*/
#define hipHostRegisterIoMemory 0x4

/** This flag is ignored On AMD devices.*/
#define hipHostRegisterReadOnly 0x08

/** Coarse Grained host memory lock.*/
#define hipExtHostRegisterCoarseGrained 0x8