#!/bin/bash

# set the env
source ../setUpModules.sh

make -f Makefile realclean
make -f Makefile

ldd ./hipfft_complex_3d.x
