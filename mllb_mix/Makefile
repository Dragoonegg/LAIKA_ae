obj-m += mllb_kern.o
mllb_kern-objs := consts.o helpers.o main.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O0 -march=native -mhard-float -msse 

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

HIP_PATH ?= $(shell \
	if [ -n "$$HIP_PATH" ]; then \
		echo "$$HIP_PATH"; \
	elif command -v hipcc >/dev/null 2>&1; then \
		dirname $$(dirname $$(readlink -f $$(command -v hipcc))); \
	elif [ -d "/opt/rocm" ]; then \
		ls -d /opt/rocm* 2>/dev/null | head -1; \
	else \
		echo "/opt/rocm"; \
	fi)
HIP_INCLUDE = -I$(HIP_PATH)/include -I$(HIP_PATH)/include/hip
HIP_LIB = -L$(HIP_PATH)/lib -lamdhip64
HIP_FLAGS = -D__HIP_PLATFORM_AMD__ -x hip -std=c++17 -fPIC -msse -msse2 -msse3 -msse4 -O0
HIPFLAGS += -ffast-math -munsafe-fp-atomics \
            -march=gfx1103 \
            -fno-builtin

all: uspace
	make -f Makefile_hsaco
	make -f Makefile_cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest
	rm -f mllb.cubin
	rm -f mllb.hsaco

# uspace:
# 	hipcc $(HIP_INCLUDE) $(HIP_FLAGS) consts.c main.c helpers.c kernels.cpp -o utest $(HIP_LIB)

cubin:
	make -f Makefile_cubin

hsaco:
	make -f Makefile_hsaco

.PHONY: uspace cubin hsaco clean
