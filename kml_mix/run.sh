#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

sudo insmod kml.ko hsaco_path=$(readlink -f ./kml.hsaco) cubin_path=$(readlink -f ./kml.cubin)
sudo rmmod kml
