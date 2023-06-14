#!/bin/bash

# set the env
#source setUpModules.sh

make -f Makefile clean
make -f Makefile

ldd ./hipfft.x
