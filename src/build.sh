#!/bin/bash

# set the env
#source setUpModules.sh

make -f Makefile realclean
make -f Makefile

export INSTALL_DIR=/home/ac.nichols/H4I/INSTALLATION/H4I-HIPFFT
export INCLUDE_DIR=${INSTALL_DIR}/include
export LIB_DIR=${INSTALL_DIR}/lib

if [ ! -d "${INSTALL_DIR}" ]; then
    mkdir ${INSTALL_DIR}
fi
if [ ! -d "${INCLUDE_DIR}" ]; then
    mkdir ${INCLUDE_DIR}
fi
if [ ! -d "${LIB_DIR}" ]; then
    mkdir ${LIB_DIR}
fi

make -f Makefile install
