#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import re
import tempfile
import pwd
import grp


# 需要保留的4组前缀
PREFIXES = (
    "KML_CPU_batch_",
    "KML_dGPU_batch_",
    "KML_APU_PL_batch_",
    "KML_APU_PK_batch_",
)


def set_file_permissions(file_path):
    """设置文件权限，让普通用户也能读取（如果通过 sudo 运行）"""
    try:
        # 如果通过 sudo 运行，获取原始用户
        original_user = os.environ.get("SUDO_USER")
        if original_user:
            try:
                user_info = pwd.getpwnam(original_user)
                uid = user_info.pw_uid
                gid = user_info.pw_gid
                # 修改文件所有者为原始用户
                os.chown(file_path, uid, gid)
                # 设置权限为 644 (rw-r--r--)
                os.chmod(file_path, 0o644)
                print(f"[INFO] Set file ownership to {original_user} and permissions to 644")
            except (KeyError, OSError) as e:
                print(f"[WARN] Failed to set file ownership: {e}", file=sys.stderr)
        else:
            # 如果没有 SUDO_USER，至少设置权限让其他用户可读
            os.chmod(file_path, 0o644)
            print(f"[INFO] Set file permissions to 644")
    except OSError as e:
        print(f"[WARN] Failed to set file permissions: {e}", file=sys.stderr)


def process_log(fin, fout):
    """处理日志：过滤、去掉时间戳、丢弃warmup样本"""
    # 记录每一组是否已经丢掉过一个 batch=16 的样本
    warmup_dropped = {p: False for p in PREFIXES}

    for raw_line in fin:
        line = raw_line.rstrip("\n")

        # 1) 过滤和输出无关的行：只保留包含任意一个前缀的行
        if not any(p in line for p in PREFIXES):
            continue

        # 2) 去掉时间戳部分：取 ']' 之后的内容
        #    例如：
        #    "[一 12月  1 19:21:45 2025] MLLB_CPU_batch_16, 14, 14"
        #    -> "MLLB_CPU_batch_16, 14, 14"
        if "] " in line:
            payload = line.split("] ", 1)[1].strip()
        else:
            # 如果意外没有时间戳，就直接用整行
            payload = line.strip()

        # 3) 判断属于哪个前缀
        prefix = None
        for p in PREFIXES:
            if payload.startswith(p):
                prefix = p
                break
        if prefix is None:
            # 理论上不会到这一步，但保险起见
            continue

        # 4) 提取 batch size
        #    匹配 "_batch_数字"
        m = re.search(r"_batch_(\d+)", payload)
        if m:
            batch = int(m.group(1))
            # 5) 丢掉每一组第一个 batch=16 的样本（warmup）
            if batch == 16 and not warmup_dropped[prefix]:
                warmup_dropped[prefix] = True
                # 不输出这一行
                continue

        # 6) 输出处理后的有效行（无时间戳）
        print(payload, file=fout)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} '<workload_cmd>' <kernel_log_path>")
        print(f"Example: {sys.argv[0]} './run_bench.sh' ./kernel.log")
        sys.exit(1)

    workload_cmd = sys.argv[1]
    log_path = sys.argv[2]

    # 建议要求以 root 运行（AE 文档里说明）
    if os.geteuid() != 0:
        print("[ERROR] This script must be run as root (for dmesg).", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Clearing previous kernel messages (dmesg -C)...")
    try:
        subprocess.run(["dmesg", "-C"], check=False)
    except FileNotFoundError:
        print("[ERROR] 'dmesg' command not found. Please install util-linux.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Starting dmesg -w -T, logging to {log_path} ...")
    try:
        log_file = open(log_path, "w")
    except OSError as e:
        print(f"[ERROR] Failed to open log file {log_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # -T: 打印可读时间戳，更方便评审看
    dmesg_proc = subprocess.Popen(
        ["dmesg", "-w", "-T"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # 稍微等一下，确保 dmesg -w 已经起来
    time.sleep(0.5)

    print(f"[INFO] Running workload command: {workload_cmd}")
    # shell=True 方便传复杂命令，AE 文档里注明不要给不可信输入
    workload_ret = subprocess.call(workload_cmd, shell=True)

    print(f"[INFO] Workload finished with exit code {workload_ret}.")
    print("[INFO] Waiting a bit to flush remaining kernel logs...")
    time.sleep(1.0)

    print("[INFO] Stopping dmesg logger...")
    dmesg_proc.terminate()
    try:
        dmesg_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print("[WARN] dmesg did not exit in time, killing...")
        dmesg_proc.kill()

    log_file.close()
    # 设置文件权限，让普通用户也能读取
    set_file_permissions(log_path)
    print(f"[INFO] Kernel log saved to {log_path}")

    # 后处理日志：过滤、去掉时间戳、丢弃warmup样本
    print(f"[INFO] Processing log file...")
    try:
        # 读取原始日志
        with open(log_path, "r", encoding="utf-8", errors="ignore") as fin:
            # 使用临时文件保存处理后的内容
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=os.path.dirname(log_path) or ".") as tmp_out:
                tmp_path = tmp_out.name
                process_log(fin, tmp_out)
        
        # 用处理后的文件覆盖原文件
        os.replace(tmp_path, log_path)
        # 再次设置权限（因为替换后的文件可能权限不对）
        set_file_permissions(log_path)
        print(f"[INFO] Log processed and saved to {log_path}")
    except Exception as e:
        print(f"[WARN] Failed to process log file: {e}", file=sys.stderr)
        print(f"[WARN] Original log file is still available at {log_path}", file=sys.stderr)

    # 把工作负载的返回码传递出去
    sys.exit(workload_ret)


if __name__ == "__main__":
    main()
