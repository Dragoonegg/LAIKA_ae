ccflags-y +=  -I$(src)/../../kapi/include
obj-$(CONFIG_ECRYPT_FS) += lake_gcm.o

lake_gcm-y := gcm_main.o gcm_hip.o 

KBUILD_EXTRA_SYMBOLS += $(src)/../../kapi/kernel/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../../kapi/kshm/Module.symvers


HIP_PATH ?= /opt/rocm-6.3.3
HIP_INCLUDE = -I$(HIP_PATH)/include -I$(HIP_PATH)/include/hip
HIP_LIB = -L$(HIP_PATH)/lib -lamdhip64
HIP_FLAGS = -D__HIP_PLATFORM_AMD__ -x hip -std=c++17 -fPIC -msse -msse2 -msse3 -msse4 -O0

#all: test_kernel test_aesni gcm_kernels.cubin
all: hip_hsaco
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f gcm_kernels.cubin
	rm -f gcm_kernels.hsaco

.PHONY: test_user test_kernel
test_user:
	nvcc test_user.c gcm_cuda.c -lcuda -o test_user

test_kernel:
	make -f Makefile_testkernel

test_aesni:
	make -f Makefile_testaesni

cubin:
	make -f Makefile_cubin

uspace:
	nvcc test_user.c gcm_cuda.c -lcuda -o test_user

hip_hsaco:	
	make -f Makefile_hip

hip_uspace:
	hipcc $(HIP_INCLUDE) $(HIP_FLAGS) test_hip.c gcm_hip.c -lcuda -o test_user $(HIP_LIB)

gcm_kernels.cubin: cubin
	make -f Makefile_cubin