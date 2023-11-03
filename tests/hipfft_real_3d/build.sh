#!/bin/bash

# set the env
source ../setUpModules.sh

make -f Makefile realclean
make -f Makefile

ldd ./hipfft_real_3d.x
