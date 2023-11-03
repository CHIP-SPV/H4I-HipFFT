#!/bin/bash

source setUpModules.sh

module list

CC=`which clang`
CXX=`which clang++`

echo $CC
echo $CXX

export BASE_DIR=${PWD}
export INSTALL_DIR="${HIPFFT_PATH}"

echo ${BASE_DIR}


if [ ! -d "H4I-HipFFT" ]; then
   git clone https://github.com/CHIP-SPV/H4I-HipFFT.git
   cd H4I-HipFFT
   git checkout develop
   cd ../
fi

cd H4I-HipFFT

if [ ! -d "build" ]; then
   mkdir build
fi

cd build

pwd

#exit


## no longer needed, but kept here for a reminder
#export MKL_DIR=$MKLROOT

## for cmake to find mklshim ...
export MKLShim_DIR="${MKLSHIM_PATH}"

cmake ../ \
      -D CMAKE_CXX_COMPILER=$CXX \
      -D USE_ONEAPI=ON \
      -D CMAKE_C_COMPILER=$CC \
      -D CMAKE_BUILD_TYPE=Release  \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} |& tee ${BASE_DIR}/my_hipfft_config

#exit

make |& tee ${BASE_DIR}/my_hipfft_build
make install |& tee ${BASE_DIR}/my_hipfft_install

