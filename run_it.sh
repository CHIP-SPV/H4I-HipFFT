#!/bin/bash

# set the env
#source setUpModules.sh

export LD_LIBRARY_PATH=/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_WITH_FFT/lib64:${LD_LIBRARY_PATH}
#ldd hipfft.x

./hipfft.x
