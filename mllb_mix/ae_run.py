#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import re
import tempfile
import pwd
import grp


# 4 groups of prefixes to keep
PREFIXES = (
    "MLLB_CPU_batch_",
    "MLLB_dGPU_batch_",
    "MLLB_APU_PL_batch_",
    "MLLB_APU_PK_batch_",
)


def set_file_permissions(file_path):
    """Set file permissions so regular users can also read (if run via sudo)"""
    try:
        # If run via sudo, get the original user
        original_user = os.environ.get("SUDO_USER")
        if original_user:
            try:
                user_info = pwd.getpwnam(original_user)
                uid = user_info.pw_uid
                gid = user_info.pw_gid
                # Change file owner to original user
                os.chown(file_path, uid, gid)
                # Set permissions to 644 (rw-r--r--)
                os.chmod(file_path, 0o644)
                print(f"[INFO] Set file ownership to {original_user} and permissions to 644")
            except (KeyError, OSError) as e:
                print(f"[WARN] Failed to set file ownership: {e}", file=sys.stderr)
        else:
            # If no SUDO_USER, at least set permissions so other users can read
            os.chmod(file_path, 0o644)
            print(f"[INFO] Set file permissions to 644")
    except OSError as e:
        print(f"[WARN] Failed to set file permissions: {e}", file=sys.stderr)


def process_log(fin, fout):
    """Process log: filter, remove timestamps, discard warmup samples"""
    # Record whether each group has already dropped one batch=16 sample
    warmup_dropped = {p: False for p in PREFIXES}

    for raw_line in fin:
        line = raw_line.rstrip("\n")

        # 1) Filter lines unrelated to output: only keep lines containing any prefix
        if not any(p in line for p in PREFIXES):
            continue

        # 2) Remove timestamp part: take content after ']'
        #    Example:
        #    "[Mon Dec  1 19:21:45 2025] MLLB_CPU_batch_16, 14, 14"
        #    -> "MLLB_CPU_batch_16, 14, 14"
        if "] " in line:
            payload = line.split("] ", 1)[1].strip()
        else:
            # If unexpectedly no timestamp, use the entire line
            payload = line.strip()

        # 3) Determine which prefix it belongs to
        prefix = None
        for p in PREFIXES:
            if payload.startswith(p):
                prefix = p
                break
        if prefix is None:
            # Theoretically shouldn't reach here, but just in case
            continue

        # 4) Extract batch size
        #    Match "_batch_number"
        m = re.search(r"_batch_(\d+)", payload)
        if m:
            batch = int(m.group(1))
            # 5) Discard the first batch=16 sample of each group (warmup)
            if batch == 16 and not warmup_dropped[prefix]:
                warmup_dropped[prefix] = True
                # Do not output this line
                continue

        # 6) Output processed valid line (no timestamp)
        print(payload, file=fout)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} '<workload_cmd>' <kernel_log_path>")
        print(f"Example: {sys.argv[0]} './run_bench.sh' ./kernel.log")
        sys.exit(1)

    workload_cmd = sys.argv[1]
    log_path = sys.argv[2]

    # Recommended to run as root (as stated in AE documentation)
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

    # -T: Print readable timestamps, more convenient for review
    dmesg_proc = subprocess.Popen(
        ["dmesg", "-w", "-T"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait a bit to ensure dmesg -w has started
    time.sleep(0.5)

    print(f"[INFO] Running workload command: {workload_cmd}")
    # shell=True convenient for passing complex commands, AE documentation notes not to give untrusted input
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
    # Set file permissions so regular users can also read
    set_file_permissions(log_path)
    print(f"[INFO] Kernel log saved to {log_path}")

    # Post-process log: filter, remove timestamps, discard warmup samples
    print(f"[INFO] Processing log file...")
    try:
        # Read original log
        with open(log_path, "r", encoding="utf-8", errors="ignore") as fin:
            # Use temporary file to save processed content
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir=os.path.dirname(log_path) or ".") as tmp_out:
                tmp_path = tmp_out.name
                process_log(fin, tmp_out)
        
        # Overwrite original file with processed file
        os.replace(tmp_path, log_path)
        # Set permissions again (because replaced file may have wrong permissions)
        set_file_permissions(log_path)
        print(f"[INFO] Log processed and saved to {log_path}")
    except Exception as e:
        print(f"[WARN] Failed to process log file: {e}", file=sys.stderr)
        print(f"[WARN] Original log file is still available at {log_path}", file=sys.stderr)

    # Pass through the workload return code
    sys.exit(workload_ret)


if __name__ == "__main__":
    main()
