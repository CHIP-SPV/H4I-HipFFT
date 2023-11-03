#!/bin/bash

source setUpModules.sh

echo $CC
echo $CXX

#exit

export BASE_DIR=${PWD}
export INSTALL_DIR="${MKLSHIM_PATH}"

echo ${BASE_DIR}

if [ ! -d "H4I-MKLShim" ]; then
   git clone https://github.com/CHIP-SPV/H4I-MKLShim.git
   cd H4I-MKLShim
   git checkout develop
   cd ../
fi

#exit

cd H4I-MKLShim

if [ ! -d "build" ]; then
   mkdir build
fi

cd build

pwd

#exit


# needed to find CL/sycl.hpp
export INCDIR="-I${CMPROOT}/linux/include/sycl"

cmake ../ \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_CXX_FLAGS="${INCDIR}/" \
	-D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR} |& tee ${BASE_DIR}/my_mklshim_config

#exit

make -j4 |& tee ${BASE_DIR}/my_mklshim_build
make install |& tee ${BASE_DIR}/my_mklshim_install

