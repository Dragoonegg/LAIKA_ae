#!/bin/bash
sudo insmod kml.ko hsaco_path=$(readlink -f ./kml.hsaco) cubin_path=$(readlink -f ./kml.cubin)
sudo rmmod kml
