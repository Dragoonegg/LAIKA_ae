# Environment Setup：

## Build the kernel：

Install required build dependencies

```bash
sudo apt-get -y install build-essential tmux git pkg-config cmake zsh
sudo apt-get install libncurses-dev gawk flex bison openssl libssl-dev dkms libelf-dev libiberty-dev autoconf zstd
sudo apt-get install libreadline-dev binutils-dev libnl-3-dev
sudo apt-get install ecryptfs-utils cpufrequtils
```

Download the kernel source

```bash
git clone https://github.com/Dragoonegg/LAKE-linux-6.0
```

Compile the kernel

```bash
cd LAKE-linux-6.0  # enter the directory
./full_compilation.sh # compile the kernel (takes about 15–20 minutes on my machine)
```

## Boot into the target kernel：

If your computer is connected to a monitor, you can select the kernel entry (6.0.0-lake) in GRUB at boot time, and edit /etc/default/grub to set (or add, if missing) the following:

```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash cma=128M@0-4G"
GRUB_CMDLINE_LINUX="cma=128M@0-4G"
```

can also automate this step using the provided script:

```bash
sudo ./setup_grup
```

Note: If no display driver is installed before this step, the system may boot into the LAKE kernel with a black screen (commonly seen on laptops after a fresh Ubuntu install). This does not affect functionality, and the issue should be resolved after installing the AMD driver. You can avoid this by installing AMD ROCm/driver ahead of time. Alternatively, select the original kernel in GRUB to boot normally.

## Install AMD Driver + ROCm

Run the following commands:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb
sudo dpkg -i amdgpu-install_6.3.60303-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video $LOGNAME
reboot                  #reboot is required to complete AMD driver installation
```

After installation, use amd-smi and hipcc --version to verify that the installation succeeded.

## Install NVIDIA Driver + CUDA

Run the following commands:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run --toolkit --driver --silent
```

After installation, use nvidia-smi and nvcc --version to verify that the installation succeeded. If nvcc is not found, you can fix the environment by running Desktop/setup_nvcc_env.sh.

# Testing：

## Root or sudo?

Because the experiments involve editing/inserting kernel modules, almost everything requires superuser. We recommend switching to the root account directly.

## Start the shared-memory and forwarding service：

```bash
git clone https://github.com/Dragoonegg/LAIKA_ae.git
cd laika_ae/kapi  	# enter laika/kapi
sudo make         # build
sudo ./load.sh 	# start the service
```

## Stop the service after all tests finish：

```bash
############# When all tests are complete, press Ctrl+C to interrupt
sudo ./unload.sh  	# stop and clean up
```

## Running Experiments：

All experiments follow a similar workflow: build → run script and collect output → extract output and plot. You can also run sudo dmesg -w to observe kernel logs.

### MLLB experiment：

```bash
cd laika_ae/mllb_mix  						# enter mllb_mix
sudo make          							# build test binaries
sudo python3 ae_run.py "./run.sh" "./kernel.log"     # take about 2 minutes
python3 ae_draw.py 						# extract data and plot
```
sudo python3 ae_run.py "./run.sh" "./kernel.log" automatically runs run.sh, captures the corresponding kernel logs, and saves them to kernel.log in the current directory. Evaluators may also use dmesg -w to capture kernel logs manually.
python3 ae_draw.py reads kernel.log, plots the figures, and saves the output as result.pdf in the current directory. Note: since this test involves CPU, NVIDIA GPU, and AMD APU, it is unrealistic to reproduce exactly the same numbers across different platforms, but the trends should be thesame.


### KML experiment：

Similar to the MLLB experiment:

```bash
cd laika_ae/kml_mix  						# enter kml_mix
sudo make								# build test binaries
sudo python3 ae_run.py "./run.sh" "./kernel.log"  # run (about 2 minutes)
python3 ae_draw.py						# extract data and plot					
```

### LinnOS experiment：

Similar to the MLLB experiment:

```bash
cd laika_ae/linnos_mix  				# enter linnos_mix
sudo make							# build test binaries
sudo python3 ae_run.py "./run.sh" "./kernel.log"  # run (about 3 minutes)
python3 ae_draw.py					# extract data and plot
```

### eCryptfs experiment：

Evaluating the accelerated eCryptfs is more complex than the previous experiments. This experiment needs to build four components: the modified eCryptfs-AMD/NV kernel module, the GCM crypto module, and the file benchmark program.

We provide a script (run.sh) that loops over each data point by executing “load module → run experiment → unload module”. The script requires a directory for creating an encrypted filesystem. On our test platform, you can create it as follows：

```bash
sudo mkfs.ext4 /dev/nvme1n1     # format the entire partition as ext4
sudo mkdir -p /mnt/nvme1        # create a mount point
sudo mount /dev/nvme1n1p1 /mnt/nvme1
sudo mkdir /mnt/nvme1/crypto    # directory for mounting eCryptfs
```

Warning: If you run on your own machine, you must carefully identify the correct device/directory (e.g., via lsblk). A mistake can cause data loss.

Next, go to laika/ecryptfs/benchmark, which contains the run.sh used in this test, but do not run it yet. It needs a directory to create encrypted subdirectories and run the file-operation benchmark. We recommend using a non-root partition formatted as ext4. After selecting the target directory, you must determine the block device name for the filesystem containing that directory (use sudo lsblk).

For example, if the directory is on the disk mounted at /home, then the device that backs /home is the target. Open run.py and modify DRIVE and ROOT_DIR

- **DRIVE**：the suffix of the block device name, without the /dev/ prefix (e.g., nvme1n1).
- **ROOT_DIR**：the absolute path of the target directory to test (it must be within the filesystem mounted from DRIVE).

This allows the script to correctly modify the readahead parameter under /sys/block/DRIVE. In the example above, ROOT_DIR is /mnt/nvme1/crypto and DRIVE is nvme1n1.

Run the experiment：

```bash
cd laika_ae/ecryptfs             # enter ecryptfs
sudo make                     # build the 4 components required by the test
cd laika_ae/ecryptfs/benchmark   # enter benchmark directory
sudo python3 run.py           # start the test (take about 10 minutes)
python3 ae_draw.py            # extract data and plot
```

The results will be saved automatically as result_mix.pdf and result_raw.pdf under the benchmark directory.

If you see the following error:

```bash
Unable to link the KEY_SPEC_USER_KEYRING into the KEY_SPEC_SESSION_KEYRING
```

This indicates a known eCryptfs bug(1377924). You can retry with a different shell, e.g., a regular shell.

If the test appears to be stuck (a full run takes ~10 minutes), you can reduce the number of test cases executed per run; for example, comment out unnecessary entries in tests and keep only the target test.
