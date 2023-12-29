#!/usr/bin/bash

#module load oneapi/eng-compiler/2023.05.15.007
#module load oneapi/eng-compiler/2023.10.15.002

module load oneapi/eng-compiler/2023.05.15.006

module use /home/pvelesko/modulefiles/

module load HIP/chipStar/llvm16/1.0/release
module load clang/clang16-spirv-omp

module load cmake

## set paths to installations
export PATH_BASE="/home/nicholsds/FROM_CHIPSTAR/INSTALLATION"
export MKLSHIM_PATH="${PATH_BASE}/MKLSHIM_DEVELOP"
export HIPFFT_PATH="${PATH_BASE}/H4I-HIPFFT"
