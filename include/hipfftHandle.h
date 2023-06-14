#pragma once

#include "h4i/mklshim/mklshim.h"

// pulled from hipfft.cpp in the repo and re-defined for 
// h4i-hipfft

struct hipfftHandle_t
{
    void* workBuffer = nullptr;
    size_t workBufferSize = 0;
    bool autoAllocate = true;
    bool workBufferNeedsFree = false;

    double scale_factor = 1.0;

    H4I::MKLShim::Context *ctxt = nullptr;
};
