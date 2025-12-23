#!/bin/bash
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

sudo insmod mllb_kern.ko hsaco_path=$(readlink -f ./mllb.hsaco) cubin_path=$(readlink -f ./mllb.cubin)
sudo rmmod mllb_kern
