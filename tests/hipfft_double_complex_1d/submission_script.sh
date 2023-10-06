#!/usr/bin/bash

#PBS -A CSC251HIAI05_CNDA 
#PBS -q debug
##PBS -q workq
#PBS -l select=1
#PBS -l walltime=05:00
##PBS -l filesystems=home
#PBS -N hipfft1d_double_complex

#echo $PWD
cd /home/nicholsds/FROM_CHIPSTAR/H4I-HipFFT/tests/hipfft_double_complex_1d
#echo $PWD

source ./setUpModules.sh
module list

exec=./hipfft_double_complex_1d.x

#ldd ${exec}

${exec}