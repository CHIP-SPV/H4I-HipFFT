#!/bin/bash

# set the env
#source setUpModules.sh

#export LD_LIBRARY_PATH=/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_WITH_FFT/lib64:/home/ac.nichols/H4I/INSTALLATION/H4I-HIPFFT/lib:${LD_LIBRARY_PATH}

make -f Makefile realclean
make -f Makefile

ldd ./hipfft_real_1d_batch.x
