#ifndef __GCM_H__
#define __GCM_H__

#ifdef __KERNEL__
#include <linux/kernel.h>
#include <linux/module.h>
#include <hip_runtime_api_mini.h>
#else
#include <stdio.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
#define PAGE_SIZE 4096u
#endif


//#define LAKE_PRINT_DEBUG

#ifdef __KERNEL__

#ifdef LAKE_PRINT_DEBUG
        #define DBG_PRINT(...) do { printk(KERN_ERR __VA_ARGS__); } while (0)
#else
	#define DBG_PRINT(...) (void)0
#endif

#define PRINT(...) do { printk(KERN_ERR __VA_ARGS__); } while (0)

#else
#define PRINT(...) do { printf(__VA_ARGS__); } while (0)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#endif

typedef unsigned long long int u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

static inline void gpuAssert(hipError_t code, const char *file, int line)
{
   if (code != hipSuccess) 
   {
#ifdef __KERNEL__
        printk(KERN_ERR "GPUassert error: %d %s %d\n", code, file, line);
#else
        const char* errs = 0;
        hipDrvGetErrorString(code, &errs);
        fprintf(stderr,"GPUassert: %s %s %d\n", errs, file, line);
        exit(code);
#endif
   }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define Nb 4
#define Nk 8
#define Nr 14

//#define FIXED_SIZE_FULL (0x1UL << 20) // 1 MB

#define SBOX_SIZE 256
#define RCON_SIZE 11
#define AESGCM_BLOCK_SIZE 16

#define AESGCM_KEYLEN 32u
#define AES_ROUNDKEYLEN 240u
#define AES_BLOCKLEN 16u
#define AES_MACLEN 12u
#define AES_GCM_STEP 64u

//const int annoying_gcc_one = 1;
//int kBaseThreadBits = 8;
//int kBaseThreadNum  = annoying_gcc_one << kBaseThreadBits;

#define kBaseThreadBits 8
#define kBaseThreadNum (1<< kBaseThreadBits)

#define crypto_aead_aes256gcm_NPUBBYTES 12U
#define crypto_aead_aes256gcm_ABYTES 16U

typedef u8 state_t[4][4];

struct AES_GCM_engine_ctx {
    hipDeviceptr_t sbox;
    hipDeviceptr_t rsbox;
    hipDeviceptr_t Rcon;
    hipDeviceptr_t key;
    hipDeviceptr_t aes_roundkey;
    hipDeviceptr_t gcm_h;

    hipDeviceptr_t HL;
    hipDeviceptr_t HH;
    hipDeviceptr_t HL_long;
    hipDeviceptr_t HH_long;
    hipDeviceptr_t HL_sqr_long;
    hipDeviceptr_t HH_sqr_long;
    hipDeviceptr_t gf_last4;
    hipDeviceptr_t nonce_device;

    hipDeviceptr_t buffer1;
    hipDeviceptr_t buffer2;

 
    hipDeviceptr_t d_src;
    hipDeviceptr_t d_dst;
    
    void* h_dst_mapped;
    void* h_src_mapped;
    hipDeviceptr_t d_dst_mapped;
    hipDeviceptr_t d_src_mapped;
   // hipDeviceptr_t d_dst_mapped;
    //XXX
    //u8 key[AESGCM_KEYLEN];
    //u8 nonce_host[crypto_aead_aes256gcm_NPUBBYTES];

    hipDevice_t device;
    hipCtx_t context;
    hipModule_t module;
    hipFunction_t xcrypt_kernel;
    hipFunction_t mac_kernel;
    hipFunction_t final_mac_kernel;
    hipFunction_t key_expansion_kernel;
    hipFunction_t setup_table_kernel;
    hipFunction_t encrypt_oneblock_kernel;
    hipFunction_t next_nonce_kernel;
    hipStream_t *g_stream;
};

void lake_AES_GCM_alloc_pages(hipDeviceptr_t* src, u32 size);
void lake_AES_GCM_free(hipDeviceptr_t src);
void lake_AES_GCM_copy_to_device(hipDeviceptr_t src, u8* buf, u32 size);
void lake_AES_GCM_copy_from_device(u8* buf, hipDeviceptr_t src, u32 size);
void lake_AES_GCM_encrypt(struct AES_GCM_engine_ctx* d_engine, hipDeviceptr_t d_dst, hipDeviceptr_t d_src, u32 size);
void lake_AES_GCM_decrypt(struct AES_GCM_engine_ctx* d_engine, hipDeviceptr_t d_dst, hipDeviceptr_t d_src, u32 size);
void lake_AES_GCM_init(struct AES_GCM_engine_ctx* d_engine);
int  lake_AES_GCM_init_fns(struct AES_GCM_engine_ctx *d_engine, char *hsaco_path);
void lake_AES_GCM_setkey(struct AES_GCM_engine_ctx* d_engine, const u8* key);
void lake_AES_GCM_destroy(struct AES_GCM_engine_ctx* d_engine);

#ifdef __HIPCC__
#define ENDIAN_SELECTOR 0x00000123
#define GETU32(plaintext) __byte_perm(*(u32*)(plaintext), 0, ENDIAN_SELECTOR)
#define PUTU32(ciphertext, st) {*(u32*)(ciphertext) = __byte_perm((st), 0, ENDIAN_SELECTOR);}
#endif
#endif
