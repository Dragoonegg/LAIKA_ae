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
#include <crypto/gf128mul.h>
#include <crypto/internal/aead.h>
#include <crypto/internal/skcipher.h>
#include <crypto/internal/hash.h>
#include <crypto/null.h>
#include <linux/scatterlist.h>
#include <crypto/gcm.h>
#include <crypto/hash.h>
//#include "internal.h"
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/fs.h> 
#include <asm/uaccess.h>
#include "lake_shm.h"
#include "gcm_hip.h"

static char *cubin_path = "gcm_kernels.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to gcm_kernels.cubin");

static char *hsaco_path = "gcm_kernels.hsaco";
module_param(hsaco_path, charp, 0444);
MODULE_PARM_DESC(hsaco_path, "The path to gcm_kernels.hsaco");

static int aesni_fraction = 0;
module_param(aesni_fraction, int, 0444);
MODULE_PARM_DESC(aesni_fraction, "Fraction of the file to be encrypted using AES-NI (out of 100), default 0");

//tfm ctx
struct crypto_gcm_ctx {
	struct AES_GCM_engine_ctx cuda_ctx;
	struct crypto_aead *aesni_tfm;
};

struct extent_crypt_result {
	struct completion completion;
	int rc;
};

static int get_aesni_fraction(int n) {
	return n*aesni_fraction/100;
}

static void extent_crypt_complete(struct crypto_async_request *req, int rc)
{
	struct extent_crypt_result *ecr = req->data;
	if (rc == -EINPROGRESS)
		return;
	ecr->rc = rc;
	DBG_PRINT("completing.. \n");
	complete(&ecr->completion);
}

static int crypto_gcm_setkey(struct crypto_aead *aead, const u8 *key,
			     unsigned int keylen)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(aead);
	int err = 0;

	if (keylen != AESGCM_KEYLEN) {
		printk(KERN_ERR "Wrong key size.. need %u, got %u\n", AESGCM_KEYLEN, keylen);
		goto out;
	}

	lake_AES_GCM_setkey(&ctx->cuda_ctx, key);
	err = crypto_aead_setkey(ctx->aesni_tfm, key, 32);
	if (err) {
		printk(KERN_ERR "err setkey\n");
		return err;
	}
out:
	return err;
}

static int crypto_gcm_setauthsize(struct crypto_aead *tfm,
				  unsigned int authsize)
{
	switch (authsize) {
	case 16:
		break;
	case 4:
	case 8:
	case 12:
	case 13:
	case 14:
	case 15:
	default:
		return -EINVAL;
	}

	return 0;
}

static int crypto_gcm_encrypt(struct aead_request *req)
{
	// static int run_times=0;
	// run_times++;
	// PRINT("run_times: %d\n", run_times);
	// when we get here, all fields in req were set by aead_request_set_crypt
	// ->src, ->dst, ->cryptlen, ->iv
	static int print_once = 0; 
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	struct aead_request **aead_req = NULL;
	int count_dst = 0, lake_count = 0;
	void* buf;
	struct scatterlist *src_sg = req->src;
	struct scatterlist *dst_sg = req->dst;
	// hipDeviceptr_t d_src = ctx->cuda_ctx.d_src;
	// hipDeviceptr_t d_dst = ctx->cuda_ctx.d_dst;
	char *pages_buf, *bad_iv,*h_buffer;
	int npages, i;
	int *rcs;
	int aesni_n, lake_n;
	struct extent_crypt_result *ecrs;

	npages = sg_nents(src_sg);
	if (sg_nents(dst_sg) != 2*npages) {
		printk(KERN_ERR "encrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}

	aesni_n = get_aesni_fraction(npages);
	lake_n = npages - aesni_n;
	DBG_PRINT("encrypt: processing %d pages. %d on aesni, %d on gpu\n", npages, aesni_n, lake_n);

	if (lake_n > 0) {
		// //sourse
		// pages_buf = (char *)kava_alloc(lake_n*PAGE_SIZE);
		// hipHostRegister(pages_buf, lake_n*PAGE_SIZE,hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
		// hipHostGetDevicePointer((void**)&d_src_mapped, pages_buf, 0);
		// ctx->cuda_ctx.d_src_mapped=d_src_mapped;
		
		// //target
		// h_buffer = (char *)kava_alloc(lake_n*PAGE_SIZE);
		// hipHostRegister(h_buffer, lake_n*PAGE_SIZE,hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
		// hipHostGetDevicePointer((void**)&d_buffer, h_buffer, 0);
		// ctx->cuda_ctx.d_buffer=d_buffer;

//--------------------------------将待加密数据拷到src
		// 预计算目标缓冲区指针，减少重复计算
		char *target_buf = ctx->cuda_ctx.h_src_mapped;
	
		// 优化：使用更大的块大小进行拷贝，减少循环开销
		#define OPTIMIZED_COPY_SIZE (16 * PAGE_SIZE)  // 4页作为一个块
		int remaining_pages = npages - aesni_n;
		int full_blocks = remaining_pages / 16;
		int remaining_single = remaining_pages % 16;
		
		// 处理完整的4页块
		for(int block = 0; block < full_blocks; block++) {
			for(int j = 0; j < 16; j++) {
				buf = sg_virt(&src_sg[aesni_n + block * 16 + j]);
				memcpy(target_buf + j * PAGE_SIZE, buf, PAGE_SIZE);
			}
			target_buf += OPTIMIZED_COPY_SIZE;
			lake_count += 16;
		}
		
		// 处理剩余的单个页面
		for(int j = 0; j < remaining_single; j++) {
			buf = sg_virt(&src_sg[aesni_n + full_blocks * 16 + j]);
			memcpy(target_buf + j * PAGE_SIZE, buf, PAGE_SIZE);
			lake_count++;
		}

		

		//TODO: copy IVs, set enc to use it. it's currently constant and set at setkey
		//lake_AES_GCM_copy_to_device(d_src, pages_buf, lake_count*PAGE_SIZE);
		// PRINT("Coping to APU\n");
		// int left = lake_count*PAGE_SIZE;
    	// int max = 1024*PAGE_SIZE;
    	// hipDeviceptr_t cur = d_src;
    	// u8* cur_buf = pages_buf;
    	// while (1) {
        // if (left <= max) {
        //     hipMemcpyHtoDAsync(cur, cur_buf, left, 0);
        //     break;
        // }
        // hipMemcpyHtoDAsync(cur, cur_buf, max, 0);
        // cur = (hipDeviceptr_t)((char*)cur + max);
        // cur_buf += max;
        // left -= max;
    	// }

		
		// hipHostRegister(pages_buf, lake_n*PAGE_SIZE,hipHostRegisterMapped | hipExtHostRegisterCoarseGrained);
		// hipHostGetDevicePointer((void**)&d_src_mapped, pages_buf, 0);
		// ctx->cuda_ctx.d_src_mapped=d_src_mapped;
//--------------------------------加密，结果放在ctx->cuda_ctx.d_buffer
		lake_AES_GCM_encrypt(&ctx->cuda_ctx, ctx->cuda_ctx.d_dst_mapped, ctx->cuda_ctx.d_src_mapped, lake_count*PAGE_SIZE);
		//PRINT("Done encrypt\n");
	}

	if (aesni_n > 0) {
		// GPU is doing work, lets do AESNI now
		// GCM can't do multiple blocks in one request..
		aead_req = vmalloc(aesni_n * sizeof(struct aead_request*));
		//ignore iv for now
		bad_iv = vmalloc(12);
		ecrs = vmalloc(aesni_n * sizeof(struct extent_crypt_result));
		rcs = vmalloc(aesni_n * sizeof(int));

		for(i = 0 ; i < 12 ; i++)
			bad_iv[i] = i;

		for(i = 0 ; i < aesni_n ;i++) {
			aead_req[i] = aead_request_alloc(ctx->aesni_tfm, GFP_NOFS);
			if (!aead_req[i]) {
				printk(KERN_ERR "err aead_request_alloc\n");
				return -1;
			}
			init_completion(&ecrs[i].completion);
			aead_request_set_callback(aead_req[i],
					CRYPTO_TFM_REQ_MAY_BACKLOG | CRYPTO_TFM_REQ_MAY_SLEEP,
					extent_crypt_complete, &ecrs[i]);
			//TODO: use req->iv
			aead_request_set_crypt(aead_req[i], &src_sg[i], &dst_sg[i*2], PAGE_SIZE, bad_iv);
			aead_request_set_ad(aead_req[i], 0);
			rcs[i] = crypto_aead_encrypt(aead_req[i]);
		}
	}


	if (lake_n > 0) {
		//copy cipher back
		//lake_AES_GCM_copy_from_device(pages_buf, d_dst, lake_count*PAGE_SIZE);
		//TODO: copy back MACs
		hipDeviceSynchronize();
//将加密后的数据写回buf
		for(i = aesni_n ; i < npages ; i++) {
			// cipher sg
			buf = sg_virt(&dst_sg[i*2]);
			memcpy(buf, ctx->cuda_ctx.h_dst_mapped+(count_dst * (PAGE_SIZE+crypto_aead_aes256gcm_ABYTES)), PAGE_SIZE);
			//TODO: copy MAC
			//memcpy(buf, pages_buf+((count_dst*PAGE_SIZE) + PAGE_SIZE), crypto_aead_aes256gcm_ABYTES);
			count_dst++;
		}
	}

	if (aesni_n > 0) {
		for(i = 0 ; i < aesni_n ; i++) {
			if (rcs[i] == -EINPROGRESS || rcs[i] == -EBUSY) {
				printk(KERN_ERR "waiting for enc req %d\n", i);
				wait_for_completion(&ecrs[i].completion);
			} 
			else if (rcs[i] == 0 || rcs[i] == -EBADMSG) {
				//ignore
			} 
			else {
				printk(KERN_ERR "decrypt error: %d\n", rcs[i]);
				return -1;
			} 
			aead_request_free(aead_req[i]);
		}
		vfree(rcs);
		vfree(aead_req);
		vfree(bad_iv);
		vfree(ecrs);
	}
	return 0;
}

static int crypto_gcm_decrypt(struct aead_request *req)
{	
	int check_blocks=5;
	static int check_once = 0;
	static int print_once = 0; 
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	struct aead_request **aead_req = NULL;
	int count_dst = 0, lake_count = 0;
	void* buf;
	struct scatterlist *src_sg = req->src; 
	struct scatterlist *dst_sg = req->dst; 
	CUdeviceptr d_src = ctx->cuda_ctx.d_src;
	CUdeviceptr d_dst = ctx->cuda_ctx.d_dst;
	char *pages_buf, *bad_iv;
	int npages, i;
	int *rcs;
	int aesni_n, lake_n;
	struct extent_crypt_result *ecrs;

	npages = sg_nents(src_sg);
	if (2*sg_nents(dst_sg) != npages) {
		printk(KERN_ERR "decrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}
	npages = npages/2;

	aesni_n = get_aesni_fraction(npages);
	lake_n = npages - aesni_n;
	DBG_PRINT("decrypt: processing %d pages. %d on aesni" "%d on gpu\n", npages, aesni_n, lake_n);

	if (lake_n > 0) {
		//pages_buf = (char *)kava_alloc(lake_n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
		// if(!pages_buf) {
		// 	printk(KERN_ERR "decrypt: error allocating %ld bytes\n", 
		// 		lake_n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
		// 	return -1;
		// }

		char *target_buf = ctx->cuda_ctx.h_src_mapped;
		
		// 优化：使用更大的块大小进行拷贝，减少循环开销
		// 注意：这里使用i*2的索引模式，所以需要调整块大小计算
		#define OPTIMIZED_COPY_SIZE_DECRYPT (16 * PAGE_SIZE)  // 16页作为一个块
		int remaining_pages = npages - aesni_n;
		int full_blocks = remaining_pages / 16;
		int remaining_single = remaining_pages % 16;
		
		// 处理完整的16页块
		for(int block = 0; block < full_blocks; block++) {
			for(int j = 0; j < 16; j++) {
				int actual_index = aesni_n + block * 16 + j;
				buf = sg_virt(&src_sg[actual_index * 2]);  // 注意i*2的索引模式
				memcpy(target_buf + j * PAGE_SIZE, buf, PAGE_SIZE);
				//TODO: copy MACs sg_virt(&src_sg[actual_index * 2 + 1]);
			}
			target_buf += OPTIMIZED_COPY_SIZE_DECRYPT;
			lake_count += 16;
		}
		
		// 处理剩余的单个页面
		for(int j = 0; j < remaining_single; j++) {
			int actual_index = aesni_n + full_blocks * 16 + j;
			buf = sg_virt(&src_sg[actual_index * 2]);  // 注意i*2的索引模式
			memcpy(target_buf + j * PAGE_SIZE, buf, PAGE_SIZE);
			//TODO: copy MACs sg_virt(&src_sg[actual_index * 2 + 1]);
			lake_count++;
		}
		
		// PRINT("decrypt: done copying to SHAREDMEM\n");
		// //lake_AES_GCM_copy_to_device(d_src, pages_buf, lake_n*PAGE_SIZE);
		// int left = lake_n*PAGE_SIZE;
    	// int max = 1024*PAGE_SIZE;
    	// hipDeviceptr_t cur = d_src;
    	// u8* cur_buf = pages_buf;
    	// while (1) {
        // if (left <= max) {
        //     hipMemcpyHtoDAsync(cur, cur_buf, left, 0);
        //     break;
        // }
        // hipMemcpyHtoDAsync(cur, cur_buf, max, 0);
        // cur = (hipDeviceptr_t)((char*)cur + max);
        // cur_buf += max;
        // left -= max;
   		// }
   		// hipStreamSynchronize(0);
		// //TODO: copy MACs too
		lake_AES_GCM_decrypt(&ctx->cuda_ctx, ctx->cuda_ctx.d_dst_mapped, ctx->cuda_ctx.d_src_mapped, lake_n*PAGE_SIZE);
		
	}

	if (aesni_n > 0) {
		// GPU is doing work, lets do AESNI now
		// GCM can't do multiple blocks in one request..
		aead_req = vmalloc(aesni_n * sizeof(struct aead_request*));
		//ignore iv for now
		bad_iv = vmalloc(12);
		ecrs = vmalloc(aesni_n * sizeof(struct extent_crypt_result));
		rcs = vmalloc(aesni_n * sizeof(int));

		for(i = 0 ; i < 12 ; i++)
			bad_iv[i] = i;

		for(i = 0 ; i < aesni_n ;i++) {
			aead_req[i] = aead_request_alloc(ctx->aesni_tfm, GFP_NOFS);
			if (!aead_req[i]) {
				printk(KERN_ERR "err aead_request_alloc\n");
				return -1;
			}
			init_completion(&ecrs[i].completion);
			aead_request_set_callback(aead_req[i],
					CRYPTO_TFM_REQ_MAY_BACKLOG | CRYPTO_TFM_REQ_MAY_SLEEP,
					extent_crypt_complete, &ecrs[i]);
			//TODO: use req->iv
			aead_request_set_crypt(aead_req[i], &src_sg[i*2], &dst_sg[i], PAGE_SIZE+crypto_aead_aes256gcm_ABYTES, bad_iv);
			aead_request_set_ad(aead_req[i], 0);
			rcs[i] = crypto_aead_decrypt(aead_req[i]);
		}
	}

	if (lake_n > 0) {
		//copy cipher back
		// lake_AES_GCM_copy_from_device(pages_buf, d_dst, lake_n*PAGE_SIZE);
		// hipStreamSynchronize(0);
		//将解密完成的数据拷回来
		hipDeviceSynchronize();
		for(i = aesni_n ; i < npages ; i++) {
			// plain sg
			buf = sg_virt(&dst_sg[i]);
			memcpy(buf, ctx->cuda_ctx.h_dst_mapped+(count_dst * PAGE_SIZE), PAGE_SIZE);
			count_dst++;
		}

			// 打印src_sg和dst_sg前check_blocks块的内容来确认结果是否一致
			// 注意：src_sg和dst_sg的对应关系是跳跃的：
			// src_sg[0]对应dst_sg[0], src_sg[2]对应dst_sg[1], src_sg[4]对应dst_sg[2], 以此类推
			// if (check_once==check_blocks*2)
			// {	
			// 	PRINT("npages:%d\n", npages);
			// 	if (npages >= 1) {
			// 		PRINT(KERN_INFO "=== 验证src_sg和dst_sg前check_blocks个块的内容 ===\n");
					
			// 		// 打印src_sg前check_blocks个块的内容（跳跃索引）
			// 		for (int j = 0; j < check_blocks; j++) {
			// 			int src_index = j * 2;  // 跳跃索引：0, 2, 4, 6, 8...
			// 			char *src_buf = sg_virt(&src_sg[src_index]);
			// 			PRINT(KERN_INFO "src_sg[%d] 前64字节: ", src_index);
			// 			for (int k = 0; k < 64 && k < PAGE_SIZE; k++) {
			// 				PRINT(KERN_CONT "%02x ", (unsigned char)src_buf[k]);
			// 			}
			// 			PRINT(KERN_CONT "\n");
			// 		}
					
			// 		// 打印dst_sg前check_blocks个块的内容（连续索引）
			// 		for (int j = 0; j < check_blocks; j++) {
			// 			char *dst_buf = sg_virt(&dst_sg[j]);
			// 			PRINT(KERN_INFO "dst_sg[%d] 前64字节: ", j);
			// 			for (int k = 0; k < 64 && k < PAGE_SIZE; k++) {
			// 				PRINT(KERN_CONT "%02x ", (unsigned char)dst_buf[k]);
			// 			}
			// 			PRINT(KERN_CONT "\n");
			// 		}
					
			// 		// 比较前check_blocks个块是否一致（使用正确的对应关系）
			// 		int blocks_match = 1;
			// 		for (int j = 0; j < check_blocks; j++) {
			// 			int src_index = j * 2;  // 跳跃索引：0, 2, 4, 6, 8...
			// 			char *src_buf = sg_virt(&src_sg[src_index]);
			// 			char *dst_buf = sg_virt(&dst_sg[j]);
			// 			PRINT(KERN_INFO "比较: src_sg[%d] vs dst_sg[%d]\n", src_index, j);
			// 			if (memcmp(src_buf, dst_buf, PAGE_SIZE) != 0) {
			// 				blocks_match = 0;
			// 				PRINT(KERN_ERR "块 %d (src_sg[%d] vs dst_sg[%d]) 内容不一致!\n", j, src_index, j);
			// 				break;
			// 			}
			// 		}
					
			// 		if (blocks_match) {
			// 			PRINT(KERN_INFO "前%d个块内容一致 ✓\n", check_blocks);
			// 		} else {
			// 			PRINT(KERN_ERR "前%d个块内容不一致 ✗\n", check_blocks);
			// 		}
			// 		PRINT(KERN_INFO "=== 验证完成 ===\n");
			// 	}
				
			// }
			// check_once++;
		

		
		
		//kava_free(pages_buf);
	}
	
	if (aesni_n > 0) {
		for(i = 0 ; i < aesni_n ; i++) {
			if (rcs[i] == -EINPROGRESS || rcs[i] == -EBUSY) {
				printk(KERN_ERR "waiting for enc req %d\n", i);
				wait_for_completion(&ecrs[i].completion);
			} 
			else if (rcs[i] == 0 || rcs[i] == -EBADMSG) {
				//ignore
			} 
			else {
				printk(KERN_ERR "decrypt error: %d\n", rcs[i]);
				return -1;
			} 
			aead_request_free(aead_req[i]);
		}
		vfree(rcs);
		vfree(aead_req);
		vfree(bad_iv);
		vfree(ecrs);
	}
	return 0;
}

static int crypto_gcm_init_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	unsigned long align;

	align = crypto_aead_alignmask(tfm);
	align &= ~(crypto_tfm_ctx_alignment() - 1);
	crypto_aead_set_reqsize(tfm, align);

	lake_AES_GCM_init_fns(&ctx->cuda_ctx, hsaco_path);
	lake_AES_GCM_init(&ctx->cuda_ctx);

	if (aesni_fraction > 100)
		aesni_fraction = 100;
	if (aesni_fraction < 0)
		aesni_fraction = 0;

	ctx->aesni_tfm = crypto_alloc_aead("generic-gcm-aesni", 0, 0);
	if (IS_ERR(ctx->aesni_tfm)) {
		printk(KERN_ERR "Error allocating generic-gcm-aesni %ld\n", PTR_ERR(ctx->aesni_tfm));
		return -ENOENT;
	}
	
	return 0;
}

static void crypto_gcm_exit_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	lake_AES_GCM_destroy(&ctx->cuda_ctx);


	crypto_free_aead(ctx->aesni_tfm);
}

static void crypto_gcm_free(struct aead_instance *inst)
{	
	kfree(inst);
}

static int crypto_gcm_create_common(struct crypto_template *tmpl,
				    struct rtattr **tb)
{
	struct aead_instance *inst;
	int err;

	err = -ENOMEM;
	inst = kzalloc(sizeof(*inst), GFP_KERNEL);
	if (!inst)
		goto out_err;

	snprintf(inst->alg.base.cra_name, CRYPTO_MAX_ALG_NAME, "lake_gcm(aes)");
	snprintf(inst->alg.base.cra_driver_name, CRYPTO_MAX_ALG_NAME, "lake(gcm_cuda,aes)");

	inst->alg.base.cra_flags = CRYPTO_ALG_ASYNC;
	//inst->alg.base.cra_priority = (ghash->base.cra_priority +
	//			       ctr->base.cra_priority) / 2;
	inst->alg.base.cra_priority = 100;
	inst->alg.base.cra_blocksize = 1;
	//XXX
	//inst->alg.base.cra_alignmask = ghash->base.cra_alignmask |
	//			       ctr->base.cra_alignmask;
	inst->alg.base.cra_alignmask = 0;
	inst->alg.base.cra_ctxsize = sizeof(struct crypto_gcm_ctx);
	inst->alg.ivsize = GCM_AES_IV_SIZE;

	//XXX
	//inst->alg.chunksize = crypto_skcipher_alg_chunksize(ctr);
	inst->alg.chunksize = 1;

	inst->alg.maxauthsize = 16;
	inst->alg.init = crypto_gcm_init_tfm;
	inst->alg.exit = crypto_gcm_exit_tfm;
	inst->alg.setkey = crypto_gcm_setkey;
	inst->alg.setauthsize = crypto_gcm_setauthsize;
	inst->alg.encrypt = crypto_gcm_encrypt;
	inst->alg.decrypt = crypto_gcm_decrypt;
	inst->free = crypto_gcm_free;
	
	err = aead_register_instance(tmpl, inst);
	if (err) {
		printk(KERN_ERR "error aead_register_instance %d\n", err);
		goto out_err;
	}
	return err;
out_err:
	printk(KERN_ERR "error in crypto_gcm_create_common %d\n", err);
	kfree(inst);
	return err;
}

static int crypto_gcm_create(struct crypto_template *tmpl, struct rtattr **tb)
{
	return crypto_gcm_create_common(tmpl, tb);
}

static struct crypto_template crypto_gcm_tmpl = {
	.name = "lake_gcm",
	.create = crypto_gcm_create,
	.module = THIS_MODULE,
};

static int __init crypto_gcm_module_init(void)
{
	int err;
	struct file *f;

	f = filp_open(hsaco_path, O_RDONLY, 0600);
	if (IS_ERR(f) || !f) {
		printk(KERN_ERR "cant open hsaco file at %s\n", hsaco_path);
		return -2;
	}
	printk(KERN_ERR "hsaco found at %s\n", hsaco_path);
	filp_close(f, 0);

	err = crypto_register_template(&crypto_gcm_tmpl);
	if (err)
		goto out_undo_gcm;
	printk(KERN_ERR "lake_gcm crypto template registered.\n");
	return 0;

out_undo_gcm:
	printk(KERN_ERR "error registering template\n");
	crypto_unregister_template(&crypto_gcm_tmpl);
	return err;
}

static void __exit crypto_gcm_module_exit(void)
{
	crypto_unregister_template(&crypto_gcm_tmpl);
}

module_init(crypto_gcm_module_init);
module_exit(crypto_gcm_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Galois/Counter Mode using CUDA");
MODULE_AUTHOR("Henrique Fingler");
MODULE_ALIAS_CRYPTO("lake_gcm");
