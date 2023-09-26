#!/bin/bash

# set the env
#source setUpModules.sh

make -f Makefile realclean
make -f Makefile

ldd ./hipfft_double_complex_1d_batch.x
