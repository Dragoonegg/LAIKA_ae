#!/bin/bash

sudo insmod lake_gcm.ko hsaco_path=$(readlink -f ./gcm_kernels.hsaco) aesni_fraction=0
