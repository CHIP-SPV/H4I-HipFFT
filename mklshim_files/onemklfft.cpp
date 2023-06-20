// Copyright 2021-2023 UT-Battelle
// See LICENSE.txt in the root of the source distribution for license info.
#include <iostream>
#include "oneapi/mkl.hpp"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"
#include "h4i/mklshim/impl/Context.h"
#include "h4i/mklshim/impl/Operation.h"


namespace H4I::MKLShim
{

#if 1

  struct fftDescriptorSR
  {
      fftDescriptorSR(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          stuff = 10.0;

	  // commit the plan
	  fft_plan.commit(ctxt->queue);

	  // wait for everything to complete before continuing
	  ctxt->queue.wait();
      }

      float stuff;
      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;
      
      ~fftDescriptorSR() {}
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

  // just a test
  void sAsum_test(Context* ctxt, int64_t n, const float *x, int64_t incx,
                  float *result) {
    // ONEMKL_TRY
    // ONEMKL_CATCH("ASUM")
  }

  void checkFFTQueue(Context* ctxt) {
     std::cout << "in checkFFTQueue : device info =  " << 
	     ctxt->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
     return;
  }

  fftDescriptorSR* createDescriptor(Context* ctxt, int64_t nx) {
     
     auto d = new fftDescriptorSR(ctxt, nx);
     std::cout << "in createDescriptor : stuff = " << d->stuff << std::endl;
     d->stuff = 4.0;
     std::cout << "in createDescriptor : stuff = " << d->stuff << std::endl;

     return d;
  }

  void checkFFTPlan(Context *ctxt, fftDescriptorSR *desc)
  {
     // ctxt->queue.wait();

     auto fft_plan = desc->fft_plan;

     int64_t value;
     fft_plan.get_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN, &value);
     std::cout << "in checkFFTPlan : FORWARD_DOMAIN = " << value << std::endl;
     fft_plan.get_value(oneapi::mkl::dft::config_param::PRECISION, &value);
     std::cout << "in checkFFTPlan : PRECISION = " << value << std::endl;
     fft_plan.get_value(oneapi::mkl::dft::config_param::DIMENSION, &value);
     std::cout << "in checkFFTPlan : DIMENSION = " << value << std::endl;
     fft_plan.get_value(oneapi::mkl::dft::config_param::LENGTHS, &value);
     std::cout << "in checkFFTPlan : LENGTHS = " << value << std::endl;
   
     return;
  }


}// end of namespace
