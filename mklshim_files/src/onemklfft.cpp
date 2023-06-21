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

  // the fft descriptors are declared in include/h4i/mklshim/onemklfft.h,
  // but since the onemklfft.h header is needed from the hipfft layer, these
  // fft descriptors are defined here since intel fft descriptors are 
  // templated with values available only in the oneapi namespace and NOT
  // from the hipfft layer
  //
  // also, structs are defined for the four possible combinations of 
  // precision (Single or Double) and starting domain (Real or Complex)

  // struct for the Single precision and Real starting domain descriptor
  struct fftDescriptorSR
  {
      // descriptor for 1d transforms
      fftDescriptorSR(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
	  // commit the plan
	  fft_plan.commit(ctxt->queue);
	  // wait for everything to complete before continuing
	  ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;
      
      ~fftDescriptorSR() {}
  };

  // struct for the Single precision and Complex starting domain descriptor
  struct fftDescriptorSC
  {
      // descriptor for 1d transforms
      fftDescriptorSC(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorSC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorSC() {}
  };

  // struct for the Double precision and Real starting domain descriptor
  struct fftDescriptorDR
  {
      // descriptor for 1d transforms
      fftDescriptorDR(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDR(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::REAL> fft_plan;

      ~fftDescriptorDR() {}
  };

  // struct for the Double precision and Complex starting domain descriptor
  struct fftDescriptorDC
  {
      // descriptor for 1d transforms
      fftDescriptorDC(Context* ctxt, std::int64_t length) : fft_plan(length)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      // descriptor for multi-dimensional transforms
      fftDescriptorDC(Context* ctxt, std::vector<std::int64_t> dimensions) : fft_plan(dimensions)
      {
          // commit the plan
          fft_plan.commit(ctxt->queue);
          // wait for everything to complete before continuing
          ctxt->queue.wait();
      }

      oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::COMPLEX> fft_plan;

      ~fftDescriptorDC() {}
  };


  // simple queue check ... will be removed
  void checkFFTQueue(Context* ctxt) {
     std::cout << "in checkFFTQueue : device info =  " << 
	     ctxt->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
     return;
  }

  // simple plan check for the SR plan ... will be removed
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

  // create the fft descriptors
  fftDescriptorSR* createFFTDescriptorSR(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorSR(ctxt, length);
     return d;
  }

  fftDescriptorSC* createFFTDescriptorSC(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorSC(ctxt, length);
     return d;
  }

  fftDescriptorDR* createFFTDescriptorDR(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorDR(ctxt, length);
     return d;
  }

  fftDescriptorDC* createFFTDescriptorDC(Context* ctxt, int64_t length) {
     auto d = new fftDescriptorDC(ctxt, length);
     return d;
  }

}// end of namespace
