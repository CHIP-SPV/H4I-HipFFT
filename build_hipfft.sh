#!/bin/bash

#source setUpModules.sh

module list

CC=`which clang`
CXX=`which clang++`

echo $CC
echo $CXX

export BASE_DIR=${PWD}
export INSTALL_DIR="/home/nicholsds/FROM_CHIPSTAR/INSTALLATION/H4I-HIPFFT"

echo ${BASE_DIR}

#cd cmake-h4i-hipfft
cd H4I-HipFFT

if [ ! -d "build" ]; then
   mkdir build
fi

cd build

pwd

#exit


export MKL_DIR=$MKLROOT
export MKLShim_DIR="/home/nicholsds/FROM_CHIPSTAR/INSTALLATION/MKLSHIM_DEVELOP"

cmake ../ \
      -D CMAKE_CXX_COMPILER=$CXX \
      -D USE_ONEAPI=ON \
      -D CMAKE_C_COMPILER=$CC \
      -D CMAKE_BUILD_TYPE=Release  \
      -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} |& tee ${BASE_DIR}/my_hipfft_config

#exit

make |& tee ${BASE_DIR}/my_hipfft_build
make install |& tee ${BASE_DIR}/my_hipfft_install

