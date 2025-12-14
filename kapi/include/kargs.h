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
#ifndef __KAPI_CUDA_UTIL_H__
#define __KAPI_CUDA_UTIL_H__

#ifdef __KERNEL__
#include <linux/ctype.h>
#include <linux/time.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/compiler.h>
#define PRINT(...) pr_err (__VA_ARGS__)
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
#else
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#define PRINT(...) printf (__VA_ARGS__)
// 用户空间：定义 likely/unlikely 宏（空操作，用于兼容性）
#ifndef likely
#define likely(x)   (x)
#endif
#ifndef unlikely
#define unlikely(x) (x)
#endif
typedef unsigned char u8;
typedef uint32_t u32;
typedef uint64_t u64;
#endif

#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_kargs_kv(void);
void destroy_kargs_kv(void);
struct kernel_args_metadata* get_kargs(const void* ptr);

#ifdef __cplusplus
}
#endif

struct kernel_args_metadata {
    int func_argc;
    size_t total_size;
    char func_arg_is_handle[64];
    size_t func_arg_size[64];
};

static inline void serialize_args(struct kernel_args_metadata* meta,
                u8* buf, void** args)
{
    int i;
    size_t size;
    // 优化：针对常见大小进行特殊处理，避免 memcpy 函数调用开销
    for (i = 0 ; i < meta->func_argc ; i++) {
        size = meta->func_arg_size[i];
        // 针对常见大小进行直接赋值，避免 memcpy 调用
        if (likely(size == sizeof(int) || size == 4)) {
            *((u32*)buf) = *((u32*)args[i]);
            buf += 4;
        } else if (likely(size == sizeof(long) || size == 8)) {
            *((u64*)buf) = *((u64*)args[i]);
            buf += 8;
        } else if (likely(size == 1)) {
            *buf = *((u8*)args[i]);
            buf += 1;
        } else {
            // 对于其他大小，使用 memcpy
            memcpy(buf, args[i], size);
            buf += size;
        }
    }
}

static inline void construct_args(struct kernel_args_metadata* meta,
                void** args, u8* buf)
{
    int i;
    for (i = 0 ; i < meta->func_argc ; i++) {
        args[i] = (void*) buf;     
// #ifndef __KERNEL__
//         printf("arg %d: %lu\n", i, meta->func_arg_size[i]);
//         if(meta->func_arg_size[i] == 8)
//             printf(" 8B:  %lx\n", *((uint64_t*) args[i]));
//         if(meta->func_arg_size[i] == 4)
//             printf(" 4B:  %x\n", *((uint32_t*) args[i]));
// #endif
        buf += meta->func_arg_size[i];
    }
}

static inline void kava_parse_function_args(const char *name, 
            struct kernel_args_metadata* meta)
{
    int *func_argc = &meta->func_argc;
    char *func_arg_is_handle = meta->func_arg_is_handle;
    size_t *func_arg_size = meta->func_arg_size;
    int i = 0, skip = 0;
    int name_len;  // 优化：缓存字符串长度，避免重复调用 strlen

    *func_argc = 0;
    if (strncmp(name, "_Z", 2)) {
        PRINT("Wrong CUDA function name");
        return;
    }

    // 优化：只计算一次字符串长度
    name_len = strlen(name);

    i = 2;
    while (i < name_len && isdigit(name[i])) {
        skip = skip * 10 + name[i] - '0';
        i++;
    }

    i += skip;
    while (i < name_len) {
        switch(name[i]) {
            case 'P':
                func_arg_size[(*func_argc)] = sizeof(CUdeviceptr);
                func_arg_is_handle[(*func_argc)++] = 1;
                //pr_info("case P, next: %c at %d\n", name[i+1], i);
                if (i + 1 < name_len &&
                        (name[i+1] == 'f' || name[i+1] == 'i' || name[i+1] == 'j' ||
                         name[i+1] == 'l' || name[i+1] == 'h' || name[i+1] == 'c' || 
                         name[i+1] == 'v' || name[i+1] == 'm'))
                    i++;
                else if (i + 1 < name_len && isdigit(name[i+1])) {
                    skip = 0;
                    while (i + 1 < name_len && isdigit(name[i+1])) {
                        skip = skip * 10 + name[i+1] - '0';
                        i++;
                    }
                    i += skip;
                }
                else {
                    PRINT("CUDA function argument: wrong pointer");
                    return;
                }
                break;

            case 'f':
            case 'i': // int
            case 'j': // unsigned int
                func_arg_size[(*func_argc)] = sizeof(int);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'l':
                func_arg_size[(*func_argc)] = sizeof(long);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'c': // char
            case 'h': // unsigned char
                func_arg_size[(*func_argc)] = sizeof(char);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'S':
                func_arg_size[(*func_argc)] = sizeof(CUdeviceptr);
                func_arg_is_handle[(*func_argc)++] = 1;
                while (i < name_len && name[i] != '_') i++;
                break;

            case 'v':
                i = name_len;
                break;

            default:
                PRINT("CUDA function argument: unrecognized type");
                return;
        }
        i++;
    }

    meta->total_size = 0;
    for (i = 0 ; i < *func_argc ; i++) {
        meta->total_size += func_arg_size[i];
    }
    //PRINT("size of args for name: %lu\n", meta->total_size);
}


#endif