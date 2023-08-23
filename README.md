# H4I-HipFFT

## Description

This is the initial version of the H4I-HipFFT layer to use hipfft on Intel GPUs. It has been tested on the JLSE Iris nodes and the ALCF Sunspot system.

## Getting started: Build H4I-MKLSHim with the following steps:

(1) Clone the develop branch of H4I-MKLShim         
(2) Copy the files in the h4i-hipfft/mklshim_files/include/h4i/mklshim and h4i-hipfft/mklshim_files/src directories 
into the H4I-MKLSHIM/include/h4i/mklshim and H4I-MKLSHIM/src directories            
(3) Source the mklshim_files/setUpModules.sh script to load the appropriate modules         
(4) Build H4I-MKLShim ... can use the mklshim_files/build_mklshim.sh script after customizing INSTALL_DIR


## Build the H4I-HipFFT library:

(1) Clone the cmake-build of this repo         
(2) Source setUpModules.sh           
(3) Edit the h4i-hipfft/build_hipfft.sh script for your environment             
(4) Run the build_hipfft.sh (the INSTALL_DIR should be customized in the step above)

## Build the test code(s) and test (current version uses make, switching to cmake soon):

(1) Source setUpModules.sh if you haven't already done so               
(2) Enter the tests directory           
(3) Enter the hipfft_real_1d directory           
(4) Edit the makefile to use the path to your H4I-MKLShim and H4I-HipFFT installations          
(5) Use the build.sh script to build the 1d test code           
(6) Use the run_it.sh code to test ... should see an error of ~1.0e-7           
(7) more tests will be added soon




