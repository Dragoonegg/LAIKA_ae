obj-m += linnos.o
linnos-objs := variables.o test_weights.o helpers.o main.o predictors.o

ccflags-y += -I$(src)/../kapi/include -I$(KAVA_ROOT)/include -I$(src)/.. -O3  -Wno-declaration-after-statement -DINFPOINT

KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kshm/Module.symvers
KBUILD_EXTRA_SYMBOLS += $(src)/../kapi/kernel/Module.symvers

all:
	make -f Makefile_cubin
	make -f Makefile_hsaco
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) modules


clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(shell pwd) clean
	rm -f utest
	rm -f linnos.cubin
	rm -f linnos.hsaco
hsaco:
	make -f Makefile_hsaco

cubin:
	make -f Makefile_cubin

.PHONY: hsaco cubin clean