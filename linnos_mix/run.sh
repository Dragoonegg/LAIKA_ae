#!/bin/bash
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi
sudo insmod linnos.ko cubin_path=$(readlink -f ./linnos.cubin) hsaco_path=$(readlink -f ./linnos.hsaco)
sudo rmmod linnos
