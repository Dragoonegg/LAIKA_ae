#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <linux/netlink.h>
#include <netlink/netlink.h>
#include <netlink/msg.h>
#include <signal.h>
#include "netlink.h"
#include "commands.h"
#include "lake_kapi.h"

static struct nl_sock *sk = NULL;

static void lake_send_cmd(uint32_t seqn, void* buf, size_t len) {
    int err;
    struct nl_msg *msg;
    struct nlmsghdr *nlh;

    msg = nlmsg_alloc_simple(MSG_LAKE_KAPI_REP, 0);
    if (!msg)
        printf("error on nlmsg_alloc_simple\n");
   
    if(buf && len) {
        err = nlmsg_append(msg, buf, len, NLMSG_ALIGNTO);
        if (err < 0)
            printf("error on nlmsg_append %d\n", err);
    }

    nl_complete_msg(sk, msg);
   
    nlh = nlmsg_hdr(msg);
    nlh->nlmsg_seq = seqn;
    err = nl_send(sk, msg);
    if(err < 0)
        printf("error on nl_send %d\n", err);

    nlmsg_free(msg);
}

static int netlink_recv_msg(struct nl_msg *msg, void *arg) {
    struct nlmsghdr *nlh;
    nlh = nlmsg_hdr(msg);
    uint32_t seq = nlh->nlmsg_seq;
    //printf("received msg with seq %u\n", seq);
    void* data = nlmsg_data(nlh);
    struct lake_cmd_ret cmd_ret;
    lake_handle_cmd(data, &cmd_ret);
    //printf("command handled, replying\n");
    if (cmd_ret.res != 0) {
        printf("CUDA API failed, returned %d\n", cmd_ret.res);
    }
    lake_send_cmd(seq, &cmd_ret, sizeof(cmd_ret));
}

void lake_destroy_socket() {
    nl_socket_free(sk);
}

void lake_recv() {
    nl_recvmsgs_default(sk);
}

int lake_init_socket() {
    int err;
    int retry_count = 0;
    const int max_retries = 10;

    sk = nl_socket_alloc();
    if (!sk) {
        fprintf(stderr, "Error allocating netlink socket\n");
        return -1;
    }
    
    nl_socket_modify_cb(sk, NL_CB_VALID, NL_CB_CUSTOM, netlink_recv_msg, NULL);
    nl_socket_disable_seq_check(sk);
    
    // Increase buffer size to improve performance
    nl_socket_set_buffer_size(sk, 2*1024*1024, 2*1024*1024);
    nl_socket_set_msg_buf_size(sk, 2*1024*1024);

    nl_socket_disable_auto_ack(sk);
    nl_socket_set_passcred(sk, 0);
    nl_socket_recv_pktinfo(sk, 0);
    nl_socket_set_nonblocking(sk);

    // Use exponential backoff strategy to reduce retry delay
    while(retry_count < max_retries) {
        err = nl_connect(sk, NETLINK_LAKE_PROT);
        if (err < 0) {
            if (retry_count == 0) {
                fprintf(stderr, "Error connecting to netlink (%d), retrying..\n", err);
            }
            // Exponential backoff: 100ms, 200ms, 400ms, 800ms, max 1s
            usleep(100000 * (1 << (retry_count < 4 ? retry_count : 4)));
            retry_count++;
        } else {
            break;
        }
    }

    if (err < 0) {
        fprintf(stderr, "Failed to connect to netlink after %d retries\n", max_retries);
        nl_socket_free(sk);
        sk = NULL;
        return -1;
    }

    //ping so kernel can get our pid
    lake_send_cmd(0, 0, 0);
    printf("Netlink connected, message sent to kernel\n");
    return 0;
}