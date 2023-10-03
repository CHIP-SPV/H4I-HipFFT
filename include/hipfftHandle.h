#pragma once

#include "h4i/mklshim/mklshim.h"

// pulled from hipfft.cpp in the hipfft repo 
// and re-defined for h4i-hipft

struct hipfftHandle_t
{
    // deal with work buffers
    void* workBuffer = nullptr;
    size_t workBufferSize = 0;
    bool autoAllocate = true;

    // are we dealing with a 2D or 3D plan??
    int Is2D = 0;
    int Is3D = 0;

    // for scaling the transforms
    double scale_factor = 1.0;

    // real starting domain (1: for r2c,c2r,d2z,z2d) or 
    // complex starting domain (0: for c2c or z2z)
    int starting_domain = 0;

    // in-place (1) or not-in-place (0) 
    int placement = 1;

    // foward (1) or inverse (0) fft
    int fft_direction = 1;

    // store the dimensions of the fft
    int64_t fft_dimensions[3];

    // store the strides for intel onemkl
    int64_t r_strides[4];
    int64_t c_strides[4];

    // for queue and device info
    H4I::MKLShim::Context *ctxt = nullptr;

    // for the unique fft plan
    H4I::MKLShim::fftDescriptorSR *descSR = nullptr;
    H4I::MKLShim::fftDescriptorSC *descSC = nullptr;
    H4I::MKLShim::fftDescriptorDR *descDR = nullptr;
    H4I::MKLShim::fftDescriptorDC *descDC = nullptr;
};
