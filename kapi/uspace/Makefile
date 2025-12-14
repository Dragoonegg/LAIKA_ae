ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

LIBS=$(shell pkg-config --libs libnl-3.0) -L/usr/local/cuda/lib64 -lcuda -lnvidia-ml
CFLAGS=$(shell pkg-config --cflags libnl-3.0) -I$(ROOT_DIR)/../include -I/usr/local/cuda/include
LIBS+=-L/opt/rocm/lib -lamdhip64 -lnvidia-ml
CFLAGS+=-I$(ROOT_DIR)/../include -I/opt/rocm/hip/include

all: lake_uspace

# kleio/py_wrapper.o:
# 	cd kleio && make

lake_uspace: *.c *.cpp
	echo compiling $^
	hipcc $(CFLAGS) $^ -o $@ $(LIBS)

clean:
	rm -f lake_uspace
