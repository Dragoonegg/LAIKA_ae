#!/bin/bash

sudo insmod mllb_kern.ko hsaco_path=$(readlink -f ./mllb.hsaco) cubin_path=$(readlink -f ./mllb.cubin)
sudo rmmod mllb_kern
