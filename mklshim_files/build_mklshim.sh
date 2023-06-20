#!/bin/bash

echo $CC
echo $CXX

export BASE_DIR=${PWD}
#export INSTALL_DIR="/home/ac.nichols/H4I/INSTALLATION/MKLSHIM/SYCL"
export INSTALL_DIR="/home/ac.nichols/H4I/INSTALLATION/MKLSHIM_WITH_FFT"

echo ${BASE_DIR}

cd master-H4I-MKLShim

if [ ! -d "build" ]; then
   mkdir build
fi

cd build

pwd

#exit


# needed to find CL/sycl.hpp
#export INCDIR="-I${CMPROOT}/linux/include -I${CMPROOT}/linux/include/sycl"
export INCDIR="-I${CMPROOT}/linux/include/sycl"

cmake ../ \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_CXX_FLAGS="${INCDIR}/" \
	-D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} |& tee ${BASE_DIR}/my_mklshim_config

#exit

make -j4 |& tee ${BASE_DIR}/my_mklshim_build
make install |& tee ${BASE_DIR}/my_mklshim_install

