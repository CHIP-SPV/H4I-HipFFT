# H4I-HipFFT

## Description

This is the initial version of the H4I-HipFFT layer to use hipfft on Intel GPUs. It has been tested on the JLSE Iris nodes.

## Getting started: Build H4I-MKLSHim with the following steps:

(1) Clone the develop branch of H4I-MKLShim         
(2) Copy the files in the mklshim_files/include/h4i/mklshim and mklshim_files/src directories 
into the H4I-MKLSHIM/include/h4i/mklshim and H4I-MKLSHIM/src directories            
(3) Source the mklshim_files/setUpModules.sh script to load the appropriate modules         
(4) Build H4I-MKLShim ... can use the mklshim_files/build_mklshim.sh script after customizing INSTALL_DIR


## Build the H4I-HipFFT library (current version uses make, switching to cmake soon):

(1) Clone this repo         
(2) Source setUpModules.sh (not necessary if you already did this when building H4I-MKLShim)            
(3) Enter the src directory         
(4) Edit the makefile to use the path to your H4I-MKLShim installation          
(5) Run the build.sh script: this will build the hipfft.so library and install the headers and library in the directory specified in the build script (the INSTALL_DIR should be customized)

## Build the test code(s) and test (current version uses make, switching to cmake soon):

(1) Enter the tests directory           
(2) Enter the hipfft_1d directory           
(3) Edit the makefile to use the path to your H4I-MKLShim and H4I-HipFFT installations          
(4) Use the build.sh script to build the 1d test code           
(5) Use the run_it.sh code to test ... should see an error of ~1.0e-7           
(6) more tests will be added soon




