// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{

  // check functions
  void checkFFTQueue(Context *ctxt);
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *descSR);
  // void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata, float *odata);
  void testFFTPlan(Context *ctxt, fftDescriptorSR *descSR, int Nbytes, float *idata);


  // create the appropriate fft descriptor
  // TO DO: need to make the integer into an array so that 
  //        I can use this functions for 1d, 2d, and 3d arrays
  fftDescriptorSR* createFFTDescriptorSR(Context *ctxt, int64_t nx);
  fftDescriptorSC* createFFTDescriptorSC(Context *ctxt, int64_t nx);
  fftDescriptorDR* createFFTDescriptorDR(Context *ctxt, int64_t nx);
  fftDescriptorDC* createFFTDescriptorDC(Context *ctxt, int64_t nx);

  void fftExecR2C(Context *ctxt, fftDescriptorSR *descSR, float *idata, float _Complex *odata);
  void fftExecC2R(Context *ctxt, fftDescriptorSR *descSR, float _Complex *idata, float *odata);
  void fftExecC2C(Context *ctxt, 
		  fftDescriptorSC *descSC, 
		  float _Complex *idata, 
		  float _Complex *odata, 
		  const int direction);

} // namespace
