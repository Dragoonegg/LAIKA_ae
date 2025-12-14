obj-m += kml.o
kml-objs := weights.o helpers.o main.o kml_cpu.o weights.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -O3 -march=native -mhard-float -msse -w

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers


HIP_PATH ?= /opt/rocm-6.3.3
HIP_INCLUDE = -I$(HIP_PATH)/include -I$(HIP_PATH)/include/hip
HIP_LIB = -L$(HIP_PATH)/lib -lamdhip64
HIP_FLAGS = -D__HIP_PLATFORM_AMD__ -x hip -std=c++17 -fPIC -msse -msse2 -msse3 -msse4 -O3 -w
HIPFLAGS += -ffast-math -munsafe-fp-atomics \
            -march=gfx1103 \
            -fno-builtin

all:
	make -B -f Makefile_hsaco
	make -B -f Makefile_cubin
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest
	rm -f kml.cubin
	rm -f kml.hsaco

hsaco:
	make -B -f Makefile_hsaco

cubin:
	make -B -f Makefile_cubin

.PHONY: uspace cubin hsaco clean