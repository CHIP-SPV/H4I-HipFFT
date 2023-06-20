#!/usr/bin/bash
#
## for JLSE h4i_hipfft
module use /soft/restricted/CNDA/modulefiles
module use /soft/modulefiles # put the appropriate modules in your path
#module use /home/pvelesko/local/modulefiles
module purge # remove any modules from your environment

#module load intel_compute_runtime # puts the latest Intel OpenCL and L0 runtimes in your environment
module load oneapi

#module load HIP/clang15/chip-spv-latest
#module load HIP/clang16/chip-spv-latest
module load chip-spv/20230531/chip-spv-release

module load cmake

