#!/bin/bash

# set the env
source ../setUpModules.sh

make -f Makefile realclean
make -f Makefile

ldd ./hipfft_real_1d.x
