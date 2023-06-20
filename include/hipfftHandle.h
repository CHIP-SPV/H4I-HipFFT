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

    // for scaling the transforms
    double scale_factor = 1.0;

    // for queue and device info
    H4I::MKLShim::Context *ctxt = nullptr;

    // for the unique fft plan
    H4I::MKLShim::fftDescriptorSR *descSR = nullptr;
    H4I::MKLShim::fftDescriptorSC *descSC = nullptr;
    H4I::MKLShim::fftDescriptorDR *descDR = nullptr;
    H4I::MKLShim::fftDescriptorDC *descDC = nullptr;
};
