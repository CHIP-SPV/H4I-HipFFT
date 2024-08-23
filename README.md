# H4I-HipFFT

H4I-HipFFT is a library that provides HipFFT functionality for Intel GPUs. It's built on top of the H4I-MKLShim library and uses Intel's oneAPI MKL for FFT operations.

## Prerequisites

- Intel oneAPI Base Toolkit
- H4I-MKLShim library (added here as a submodule)

## Building

The best way to build this library is to build it as a part of chipStar. 
- Add `-DCHIP_BUILD_HIPFFT=ON` during chipStar configure

Alternatively, this library can also be built using a pre-built chipStar
- Either set `hip_DIR` to chipStar install location or add a path to `CMAKE_PREFIX_PATH`
e.x. :

```
hip_DIR=~/space/install/HIP/chipStar/2024.07.31 cmake ../
```
or 
```
cmake ../ -DCMAKE_PREFIX_PATH=~/space/install/HIP/chipStar/2024.07.31/lib/cmake/hip
```