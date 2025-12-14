#!/bin/bash

sudo insmod linnos.ko cubin_path=$(readlink -f ./linnos.cubin) hsaco_path=$(readlink -f ./linnos.hsaco)
sudo rmmod linnos
