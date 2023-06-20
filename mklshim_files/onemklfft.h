// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
# pragma once

namespace H4I::MKLShim
{

#if 0

  struct fftDescriptorSR
  {
      float stuff;
  };

  struct fftDescriptorSC
  {
      float _Complex stuff;
  };

  struct fftDescriptorDR
  {
      double stuff;
  };

  struct fftDescriptorDC
  {
      double _Complex stuff;
  };

#endif

  void sAsum_test(Context* ctxt, int64_t n, const float *x, int64_t incx,
                  float *result);

  void checkFFTQueue(Context *ctxt);
  void checkFFTPlan(Context *ctxt, fftDescriptorSR *descSR);

  // void createDescriptor(Context *ctxt, int64_t nx);
  fftDescriptorSR* createDescriptor(Context *ctxt, int64_t nx);

} // namespace
