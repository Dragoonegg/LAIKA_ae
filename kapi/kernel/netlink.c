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
#include <linux/netlink.h>
#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/mm.h>
#include <net/sock.h>
#include <linux/xarray.h>
#include <linux/completion.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/atomic.h>
#include "netlink.h"
#include "commands.h"

static struct sock *sk = NULL;
//DEFINE_XARRAY_ALLOC(cmds_xa); 
DEFINE_XARRAY(cmds_xa); 
static struct kmem_cache *cmd_cache;
static pid_t worker_pid = -1;
static int max_counter = (1<<10);
#define MAX_COUNTER_MASK ((1<<10) - 1)  // Bit mask for fast modulo operation

// Use atomic operations instead of locks to improve concurrent performance
static atomic_t id_counter = ATOMIC_INIT(0);

// Use atomic operations to protect last_cu_err, avoid data races
static atomic_t last_cu_err = ATOMIC_INIT(0);


struct cmd_data {
    struct completion cmd_done;
    struct lake_cmd_ret ret;
    char sync;
}; //__attribute__ ((aligned (8)));


// ret is only filled in case sync is CMD_SYNC
void lake_send_cmd(void *buf, size_t size, char sync, struct lake_cmd_ret* ret)
{
    int err;
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    struct cmd_data *cmd;
    u32 xa_idx;
    CUresult cu_err;

    // Remove unnecessary locks, kmem_cache_alloc is thread-safe
    cmd = (struct cmd_data*) kmem_cache_alloc(cmd_cache, GFP_KERNEL);
    if (unlikely(!cmd)) {
        pr_warn("Error allocating from cache\n");
        ret->res = CUDA_ERROR_OUT_OF_MEMORY;
        return;
    }

    //init completion so we can wait on it
    init_completion(&cmd->cmd_done);
    cmd->sync = sync;

    // Use bitwise operations instead of modulo (max_counter is a power of 2), improve performance
    xa_idx = atomic_fetch_add(1, &id_counter) & MAX_COUNTER_MASK;

    // Store to xarray first, can clean up immediately if it fails
    err = xa_err(xa_store(&cmds_xa, xa_idx, (void*)cmd, GFP_KERNEL));
    if (unlikely(err)) {
        pr_warn("Error storing to xarray: %d\n", err);
        kmem_cache_free(cmd_cache, cmd);
        ret->res = CUDA_ERROR_OPERATING_SYSTEM;
        return;
    }

    //create netlink cmd
    skb_out = nlmsg_new(size, 0);
    if (unlikely(!skb_out)) {
        pr_err("Failed to allocate netlink skb\n");
        xa_erase(&cmds_xa, xa_idx);
        kmem_cache_free(cmd_cache, cmd);
        ret->res = CUDA_ERROR_OUT_OF_MEMORY;
        return;
    }
    
    nlh = nlmsg_put(skb_out, 0, xa_idx, MSG_LAKE_KAPI_REQ, size, 0);
    NETLINK_CB(skb_out).dst_group = 0;
    memcpy(nlmsg_data(nlh), buf, size);

    err = netlink_unicast(sk, skb_out, worker_pid, 0);
    if (unlikely(err < 0)) {
        pr_err("Failed to send netlink skb to API server, error=%d\n", err);
        nlmsg_free(skb_out);
        xa_erase(&cmds_xa, xa_idx);
        kmem_cache_free(cmd_cache, cmd);
        ret->res = CUDA_ERROR_OPERATING_SYSTEM;
        return;
    }

    // sync if requested
    if (sync == CMD_SYNC) {
        // Directly use wait_for_completion, avoid unnecessary loops
        wait_for_completion(&cmd->cmd_done);
        
        // Direct assignment instead of memcpy (small structure)
        *ret = cmd->ret;
        
        // if we sync, its like the cmd never existed, so clear every trace
        kmem_cache_free(cmd_cache, cmd);
        xa_erase(&cmds_xa, xa_idx);
        
        // For sync commands, ret->res has been copied from cmd->ret, no need to check last_cu_err
        return;
    }

    // Use atomic operations to read last_cu_err (for async commands only)
    cu_err = (CUresult)atomic_read(&last_cu_err);
    if (likely(cu_err == 0))
        ret->res = CUDA_SUCCESS;
    else
        ret->res = cu_err;
}

static void netlink_recv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = (struct nlmsghdr*) skb->data;
    //struct lake_cmd_ret *ret = (struct lake_cmd_ret*) nlmsg_data(nlh);
    void *ret = NLMSG_DATA(nlh);
    struct cmd_data *cmd;
    u32 xa_idx = nlh->nlmsg_seq;
    CUresult cu_err;

    if (unlikely(worker_pid == -1)) {
        worker_pid = nlh->nlmsg_pid;
        printk(KERN_INFO "Setting worker PID to %d\n", worker_pid);
        return;
    }

    // xa_load internally handles concurrency, no need for additional locks
    cmd = (struct cmd_data*) xa_load(&cmds_xa, xa_idx);
    if (unlikely(!cmd)) {
        pr_warn("Error (0) looking up cmd %u at xarray\n", xa_idx);
        xa_erase(&cmds_xa, xa_idx);
        return;
    }
    
    // Direct assignment instead of memcpy (small structure)
    cmd->ret = *(struct lake_cmd_ret*)ret;

    //if the cmd is async, no one will read this cmd, so clear
    if (cmd->sync == CMD_ASYNC) {
        cu_err = cmd->ret.res;
        if (unlikely(cu_err > 0)) {
            // Use atomic operations to update last_cu_err
            atomic_set(&last_cu_err, cu_err);
        }
        //erase from xarray
        xa_erase(&cmds_xa, xa_idx);
        //free from cache
        kmem_cache_free(cmd_cache, cmd);
    }
    else {
        //if there's anyone waiting, free them
        //if the cmd is sync, whoever we woke up will clean up
        complete(&cmd->cmd_done);
    }
}

static void null_constructor(void *argument) {
}

int lake_init_socket(void) {
    static struct netlink_kernel_cfg netlink_cfg = {
        .input = netlink_recv_msg,
    };

    sk = netlink_kernel_create(&init_net, NETLINK_LAKE_PROT, &netlink_cfg);
    if (!sk) {
        pr_err("Error creating netlink socket\n");
        return -ENOMEM;
    }

    //init slab cache (xarray requires at least 4-alignment)
    cmd_cache = kmem_cache_create("lake_cmd_cache", sizeof(struct cmd_data), 8, 0, null_constructor);
    if(IS_ERR(cmd_cache)) {
        pr_warn("Error creating cache: %ld\n", PTR_ERR(cmd_cache));
        return -ENOMEM;
    }
    return 0;
}

void lake_destroy_socket(void) {
    unsigned long idx = 0;
    void* entry;
    //TODO: set a halt flag

    //free up all cache entries so the kernel doesnt yell at us
    xa_for_each(&cmds_xa, idx, entry) {
        if (entry)
            kmem_cache_free(cmd_cache, entry);
    }

    //kmem_cache_destroy(cmd_cache);
    xa_destroy(&cmds_xa);
    netlink_kernel_release(sk);
}