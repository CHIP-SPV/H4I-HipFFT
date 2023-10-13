
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include <vector>
#include "hipfft.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"

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



hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    // this has been reused from H4I-HipBLAS with the following changes:
    // (1) the MKLShim context pointer ctxt is stored in the hipfftHandle plan 
    //     instead of being recast from the hipblasHandle
    // (2) Using HIPFFT_INVALID_VALUE as the returning hipfftResult since 
    //     hipfft doesn't have a NULLPTR value like hipBlas

    if (plan != nullptr)
    {
        // Obtain the underlying CHIP-SPV handles.
        // Note this code uses a CHIP-SPV extension to the HIP API.
        // See CHIP-SPV documentation for its use.
        // Both Level Zero and OpenCL backends currently require us
        // to pass nHandles = 4, and provide space for at least 4 handles.
        // TODO is there a way to query this info at runtime?
        int nHandles = H4I::MKLShim::nHandles;
        std::array<uintptr_t, H4I::MKLShim::nHandles> nativeHandles;
        hipGetBackendNativeHandles(reinterpret_cast<uintptr_t>(stream),
                nativeHandles.data(), &nHandles);

        H4I::MKLShim::SetStream(plan->ctxt, nativeHandles);
    }

    return (plan != nullptr) ? HIPFFT_SUCCESS : HIPFFT_INVALID_VALUE;
}

hipfftResult hipfftCreate(hipfftHandle *plan)
{
    // create the plan handle
    hipfftHandle h = new hipfftHandle_t;

    // HIP supports multiple backends hence query current backend name
    auto backendName = hipGetBackendName();
    // Obtain the handles to the back handlers.
    unsigned long handles[4];
    int           nHandles = 4;
    hipGetBackendNativeHandles((uintptr_t)NULL, handles, &nHandles);
    auto *ctxt = H4I::MKLShim::Create(handles, nHandles, backendName);

    // assign h->ctxt to use a consistent queue context with CHIP-SPV
    h->ctxt = ctxt;

    // assign plan and return
    *plan = h;

    // rocFFT/hipFFT returns either HIPFFT_SUCCESS or HIPFFT_INVALID_VALUE
    // So, mimic that behavior here ... unless there's a better idea
    return (*plan != nullptr) ? HIPFFT_SUCCESS : HIPFFT_INVALID_VALUE;
}

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    if(plan != nullptr)
    {
        hipError_t error = hipSuccess;

        // clean-up workBuffer
        if (plan->workBuffer != nullptr) error = hipFree(plan->workBuffer);

        // clean-up ddescriptors
        if (plan->descSR != nullptr) 
        {
           H4I::MKLShim::destroyFFTDescriptorSR(plan->ctxt,plan->descSR);
        }

        if (plan->descSC != nullptr)
        {
           H4I::MKLShim::destroyFFTDescriptorSC(plan->ctxt,plan->descSC);
        }

        if (plan->descDR != nullptr)
        {
           H4I::MKLShim::destroyFFTDescriptorDR(plan->ctxt,plan->descDR);
        }

        if (plan->descDC != nullptr)
        {
           H4I::MKLShim::destroyFFTDescriptorDC(plan->ctxt,plan->descDC);
        }

        // destroy ctxt
        H4I::MKLShim::Destroy(plan->ctxt);

        // final clean-up
        delete plan;
    }

    return HIPFFT_SUCCESS;
}


hipfftResult hipfftPlan1d(hipfftHandle *plan,
                          int nx,
                          hipfftType type,
                          int batch /* deprecated - use hipfftPlanMany */)                                        
{
    hipfftResult result = HIPFFT_SUCCESS;

    if (nx > 0)
    {
        // local handle 
        hipfftHandle h = nullptr;

	// create a new handle
        result = hipfftCreate(&h);

	// call hipfftMakePlan1d to create the descriptor
        result = hipfftMakePlan1d(h, nx, type, batch, nullptr);

        // set plan
        *plan = h;
    }
    else
    {
        *plan = nullptr;
        result = HIPFFT_INVALID_VALUE;
    }

  return result;
}




hipfftResult hipfftMakePlan1d(hipfftHandle plan,
                              int nx,
                              hipfftType type,
                              int batch /* deprecated - use hipfftPlanMany */,
			      size_t *workSize)
{
    // pre-set result
    hipfftResult result = HIPFFT_SUCCESS;

    // create the appropriate descriptor for the fft plan
    switch (type)
    {
       case HIPFFT_R2C:
       case HIPFFT_C2R:
       {
	  auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,nx);
	  plan->descSR = desc;
          break;
       }
       case HIPFFT_D2Z:
       case HIPFFT_Z2D:
       {
	  auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,nx);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_C2C:
       {
	  auto *desc = H4I::MKLShim::createFFTDescriptorSC(plan->ctxt,nx);
          plan->descSC = desc;
          break;
       }
       case HIPFFT_Z2Z:
       {
	  auto *desc = H4I::MKLShim::createFFTDescriptorDC(plan->ctxt,nx);
          plan->descDC = desc;
          break;
       }
       default:
       {
          result = HIPFFT_INVALID_PLAN;
          break;
       }
    }

    return result;
}




// will be removed
void checkPlan(hipfftHandle plan)
{
    H4I::MKLShim::checkFFTQueue(plan->ctxt);
    if (plan->descSR != nullptr) H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descSR);
    if (plan->descSC != nullptr) H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descSC);
    // if (plan->descDR != nullptr) H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descDR);
    // if (plan->descDC != nullptr) H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descDC);
    return;
}

// will be removed
void testPlan(hipfftHandle plan, int Nbytes, hipfftReal* idata, hipfftReal* odata)
{
    // H4I::MKLShim::testFFTPlan(plan->ctxt, plan->descSR, Nbytes, idata, odata);
    H4I::MKLShim::testFFTPlan(plan->ctxt, plan->descSR, Nbytes, idata);

    return;
}


hipfftResult hipfftPlan2d(hipfftHandle *plan,
                          int nx,  // slowest hipfft index
                          int ny,  // fastest hipfft index
                          hipfftType type)
{
    hipfftResult result = HIPFFT_SUCCESS;

    if ((nx > 0) && (ny > 0))
    {
        // local handle
        hipfftHandle h = nullptr;

        // create a new handle
        result = hipfftCreate(&h);

        // call hipfftMakePlan2d to create the descriptor
        result = hipfftMakePlan2d(h, nx, ny, type, nullptr);

        // set plan
        *plan = h;
    }
    else
    {
        *plan = nullptr;
        result = HIPFFT_INVALID_VALUE;
    }

    return result;
}


hipfftResult hipfftMakePlan2d(hipfftHandle plan,
                          int nx,  // slowest hipfft index
                          int ny,  // fastest hipfft index
                          hipfftType type,
                          size_t *workSize)
{
    hipfftResult result = HIPFFT_SUCCESS;

    // define the dimensions vector
    // NOTE: may need to swap these ... need to look at the intel docs, 
    //       but I think intel follows the same convention as hipfft
    std::vector<std::int64_t> dimensions {(int64_t)nx, (int64_t)ny};

    plan->Is2D = 1;

    // create the appropriate descriptor for the fft plan
    switch (type)
    {
       case HIPFFT_R2C:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);
          plan->descSR = desc;
          break;
       }
       case HIPFFT_C2R:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);
          plan->descSR = desc;
          break;
       }
       case HIPFFT_D2Z:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_Z2D:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_C2C:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSC(plan->ctxt,dimensions);
          plan->descSC = desc;
          break;
       }
       case HIPFFT_Z2Z:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDC(plan->ctxt,dimensions);
          plan->descDC = desc;
          break;
       }
       default:
       {
          result = HIPFFT_INVALID_PLAN;
          break;
       }
    }

    return result;
}

hipfftResult hipfftPlan3d(hipfftHandle *plan,
                          int nx,  // slowest hipfft index
                          int ny, 
                          int nz,  // fastest hipfft index
                          hipfftType type)
{
    hipfftResult result = HIPFFT_SUCCESS;

    if ((nx > 0) && (ny > 0) && (nz > 0))
    {
        // local handle
        hipfftHandle h = nullptr;

        // create a new handle
        result = hipfftCreate(&h);

        // call hipfftMakePlan2d to create the descriptor
        result = hipfftMakePlan3d(h, nx, ny, nz, type, nullptr);

        // set plan
        *plan = h;
    }
    else
    {
        *plan = nullptr;
        result = HIPFFT_INVALID_VALUE;
    }

    return result;
}

hipfftResult hipfftMakePlan3d(hipfftHandle plan,
                          int nx,  // slowest hipfft index
                          int ny,
                          int nz,  // fastest hipfft index
                          hipfftType type,
                          size_t *workSize)
{
    hipfftResult result = HIPFFT_SUCCESS;

    int i;

    // define the dimensions vector
    // NOTE: may need to swap these ... need to look at the intel docs, 
    //       but I think intel follows the same convention as hipfft
    std::vector<std::int64_t> dimensions {(int64_t)nx, (int64_t)ny, (int64_t)nz};

    plan->Is3D = 1;

    // create the appropriate descriptor for the fft plan
    switch (type)
    {
       case HIPFFT_R2C:
       {
          plan->starting_domain = 1;
          plan->fft_direction = 1;
          
          // set the fft dimensions ...
          // slowest changing index to fastest changing index
          for (i = 0; i < 3; i ++)
          {
            plan->fft_dimensions[i] = dimensions[i];
          }

          // assume in-place transforms and deal with the contraction in the 
          // leading (the fastest index) dimension
          int64_t c_length = (dimensions[2])/2 + 1;
          int64_t r_length = 2*c_length; // may need to be reset 

          // set the layout for physical (real) and wavenumber (complex) spaces
          int64_t in_strides[4] = {0, dimensions[1]*r_length, r_length, 1};
          int64_t out_strides[4] = {0, dimensions[1]*c_length, c_length, 1};

          // set the fft strides
          for (i = 0; i < 4; i ++) 
          {
            plan->r_strides[i] = in_strides[i];
            plan->c_strides[i] = out_strides[i];
          }

          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions,
                                                           in_strides,out_strides);
          plan->descSR = desc;

          break;
       }
       case HIPFFT_C2R:
       {
          plan->starting_domain = 1;
          plan->fft_direction = 0;

          // set the fft dimensions ...
          // slowest changing index to fastest changing index
          for (i = 0; i < 3; i ++)
          {
            plan->fft_dimensions[i] = dimensions[i];
          }

          // assume in-place transforms and deal with the contraction in the 
          // leading (the fastest index) dimension
          int64_t c_length = (dimensions[2])/2 + 1;
          int64_t r_length = 2*c_length; // may need to be reset 

          // set the layout for physical (real) and wavenumber (complex) spaces
          int64_t out_strides[4] = {0, dimensions[1]*r_length, r_length, 1};
          int64_t in_strides[4] = {0, dimensions[1]*c_length, c_length, 1};

          // set the fft strides
          for (i = 0; i < 4; i ++) 
          {
            plan->r_strides[i] = out_strides[i];
            plan->c_strides[i] = in_strides[i];
          }

          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions,
                                                           in_strides,out_strides);
          plan->descSR = desc;
          break;
       }
       case HIPFFT_D2Z:
       {
          plan->starting_domain = 1;
          plan->fft_direction = 1;

          // set the fft dimensions ...
          // slowest changing index to fastest changing index
          for (i = 0; i < 3; i ++)
          {
            plan->fft_dimensions[i] = dimensions[i];
          }

          // assume in-place transforms and deal with the contraction in the 
          // leading (the fastest index) dimension
          int64_t c_length = (dimensions[2])/2 + 1;
          int64_t r_length = 2*c_length; // may need to be reset 

          // set the layout for physical (real) and wavenumber (complex) spaces
          int64_t in_strides[4] = {0, dimensions[1]*r_length, r_length, 1};
          int64_t out_strides[4] = {0, dimensions[1]*c_length, c_length, 1};

          // set the fft strides
          for (i = 0; i < 4; i ++)
          {
            plan->r_strides[i] = in_strides[i];
            plan->c_strides[i] = out_strides[i];
          }

          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions,
                                                           in_strides,out_strides);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_Z2D:
       {
          plan->starting_domain = 1;
          plan->fft_direction = 0;

          // set the fft dimensions ...
          // slowest changing index to fastest changing index
          for (i = 0; i < 3; i ++)
          {
            plan->fft_dimensions[i] = dimensions[i];
          }

          // assume in-place transforms and deal with the contraction in the 
          // leading (the fastest index) dimension
          int64_t c_length = (dimensions[2])/2 + 1;
          int64_t r_length = 2*c_length; // may need to be reset 

          // set the layout for physical (real) and wavenumber (complex) spaces
          int64_t out_strides[4] = {0, dimensions[1]*r_length, r_length, 1};
          int64_t in_strides[4] = {0, dimensions[1]*c_length, c_length, 1};

          // set the fft strides
          for (i = 0; i < 4; i ++)
          {
            plan->r_strides[i] = out_strides[i];
            plan->c_strides[i] = in_strides[i];
          }

          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions,
                                                           in_strides,out_strides);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_C2C:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSC(plan->ctxt,dimensions);
          plan->descSC = desc;
          break;
       }
       case HIPFFT_Z2Z:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDC(plan->ctxt,dimensions);
          plan->descDC = desc;
          break;
       }
       default:
       {
          result = HIPFFT_INVALID_PLAN;
          break;
       }
    }

    return result;
}

hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch)
{
    hipfftResult result = HIPFFT_SUCCESS;

    if ((rank > 0) && (istride > 0) && (idist > 0) && 
        (ostride > 0) && (odist > 0) && (batch > 0) &&
        (n != nullptr) && (inembed != nullptr) && (onembed != nullptr))
    {
        // local handle
        hipfftHandle h = nullptr;

        // create a new handle
        result = hipfftCreate(&h);

        // call hipfftMakePlan2d to create the descriptor
        result = hipfftMakePlanMany(h, rank, n, 
                                    inembed, istride, idist, 
                                    onembed, ostride, odist, 
                                    type, batch, nullptr);

        // set plan
        *plan = h;
    }
    else
    {
        *plan = nullptr;
        result = HIPFFT_INVALID_VALUE;
    }

    return result;
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                int          rank,
                                int*         n,
                                int*         inembed,
                                int          istride,
                                int          idist,
                                int*         onembed,
                                int          ostride,
                                int          odist,
                                hipfftType   type,
                                int          batch,
                                size_t*      workSize)
{
    hipfftResult result = HIPFFT_SUCCESS;

    // full disclosure: I am not sure how to use inembed and onembed
    // and currently ignore them ... will revisit this in the future

    // also, I am letting the onemkl library handle the work buffers
    // since I've had trouble with this so far ... initially passing 
    // workSize into this function as a nullptr ... if it is NOT
    // a nullptr, then I set it to zero and return to caller as zero 
    // for now ... will revisit this in the future
    if (workSize != nullptr) *workSize = (size_t)0;

    // rank = number of fft dimensions being performed (ie. 1 = 1d, 2 = 2d, 3 = 3d)
    // and "n" is an array with "rank" entries : (ie. n[rank])
    std::vector<std::int64_t> dimensions;

    // may be able to pass "n" directly to createFFTDescriptor** instead of 
    // creating and filling "dimensions" but will use "dimensions" to control
    // the data type
    int i;
    for (i = 0; i < rank; i++)
    {
        dimensions.push_back((int64_t)n[i]);
    }

    // for (i = 0; i < rank; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     std::cout << "dimensions[" << i << "] = " << dimensions[i] << std::endl;
    // }
    // std::cout << "rank  " << rank << std::endl;
    // std::cout << "dimensions.size() = " << dimensions.size() << std::endl << std::endl;

    int64_t length = dimensions[0];
    int64_t fwd_distance, bwd_distance;
    int64_t input_stride, output_stride;
    int64_t number_of_transforms;

    // create the appropriate descriptor for the fft plan and then set plan values
    switch (type)
    {
       case HIPFFT_R2C:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);

          // set parameters for intel onemkl fft with real starting domain
          fwd_distance = (int64_t)idist;
          bwd_distance = (int64_t)odist;
          input_stride = (int64_t)istride;
          output_stride = (int64_t)ostride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesSR(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);

          plan->descSR = desc;

          break;
       }
       case HIPFFT_C2R:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);
                                
          // need to reverse the istride/ostride and idist/odist values to 
          // account for the way Intel OneMKL FFT handles the forward and 
          // backward transform parameters for an FFT with a real starting 
          // domain

          // set parameters for intel onemkl fft with real starting domain
          fwd_distance = (int64_t)odist;
          bwd_distance = (int64_t)idist;
          input_stride = (int64_t)ostride;
          output_stride = (int64_t)istride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesSR(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);

          plan->descSR = desc;
    
          break;
       }
       case HIPFFT_D2Z:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions);

          // set parameters for intel onemkl fft with real starting domain
          fwd_distance = (int64_t)idist;
          bwd_distance = (int64_t)odist;
          input_stride = (int64_t)istride;
          output_stride = (int64_t)ostride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesDR(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);
          plan->descDR = desc;
          break;
       }
       case HIPFFT_Z2D:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt,dimensions);

          // need to reverse the istride/ostride and idist/odist values to 
          // account for the way Intel OneMKL FFT handles the forward and 
          // backward transform parameters for an FFT with a real starting 
          // domain

          // set parameters for intel onemkl fft with real starting domain
          fwd_distance = (int64_t)odist;
          bwd_distance = (int64_t)idist;
          input_stride = (int64_t)ostride;
          output_stride = (int64_t)istride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesDR(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);

          plan->descDR = desc;
          break;
       }
       case HIPFFT_C2C:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSC(plan->ctxt,dimensions);

          // set parameters for intel onemkl fft with a complex starting domain
          fwd_distance = (int64_t)idist;
          bwd_distance = (int64_t)odist;
          input_stride = (int64_t)istride;
          output_stride = (int64_t)ostride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesSC(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);

          plan->descSC = desc;
          break;
       }
       case HIPFFT_Z2Z:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorDC(plan->ctxt,dimensions);

          // set parameters for intel onemkl fft with a complex starting domain
          fwd_distance = (int64_t)idist;
          bwd_distance = (int64_t)odist;
          input_stride = (int64_t)istride;
          output_stride = (int64_t)ostride;
          number_of_transforms = (int64_t)batch;

          H4I::MKLShim::setFFTPlanValuesDC(plan->ctxt, desc,
                                           input_stride, fwd_distance,
                                           output_stride, bwd_distance,
                                           number_of_transforms);

          plan->descDC = desc;
          break;
       }
       default:
       {
          result = HIPFFT_INVALID_PLAN;
          break;
       }
    }

    return result;
}

void CheckPlacement(hipfftHandle plan, void *idata, void *odata,
                    int64_t *reset_placement, int64_t *reset_r_strides)
{

    if (idata == odata) // in-place transform
      {
        if (plan->placement == 1) // plan initialized to in-place
          {
            *reset_r_strides = 0;
            *reset_placement = 0;
          }
        else // plan initialized to not-in-place and needs to be changed
          {
            if ((plan->Is2D == 1) || (plan->Is3D == 1))
              {
                // need to modify r_strides ... need to find a clean way to do this
                std::cout << "NIP2IP: original strides ..." << std::endl;
                std::cout << plan->r_strides[0] << " "
                          << plan->r_strides[1] << " "
                          << plan->r_strides[2] << " "
                          << plan->r_strides[3] << std::endl;

                int64_t r_length;
                if (plan->Is2D == 1)
                  {
                    r_length = 2*(plan->fft_dimensions[1]/2 + 1);
                    plan->r_strides[1] = r_length;
                  }
                if (plan->Is3D == 1)
                  {
                    r_length = 2*(plan->fft_dimensions[2]/2 + 1);
                    plan->r_strides[2] = r_length;
                    plan->r_strides[1] = plan->fft_dimensions[1]*r_length;
                  }

                std::cout << "NIP2IP: reset strides ..." << std::endl;
                std::cout << plan->r_strides[0] << " "
                          << plan->r_strides[1] << " "
                          << plan->r_strides[2] << " "
                          << plan->r_strides[3] << std::endl;

                *reset_r_strides = 1;
              }

            *reset_placement = 1;
            plan->placement = 1;
          }
      }
    else // not-in-place transform
      {
        if (plan->placement == 0) // plan initialized to not-in-place
          {
            *reset_placement = 0;
            *reset_r_strides = 0;
          }
        else // plan initialized to in-place and needs to be changed
          {
            if ((plan->Is2D == 1) || (plan->Is3D == 1))
              {
                // need to modify r_strides ... need to find a clean way to do this
                std::cout << "IP2NIP: original strides ..." << std::endl;
                std::cout << plan->r_strides[0] << " "
                          << plan->r_strides[1] << " "
                          << plan->r_strides[2] << " "
                          << plan->r_strides[3] << std::endl;

                if (plan->Is2D == 1)
                  {
                    plan->r_strides[1] = plan->fft_dimensions[1];
                  }
                if (plan->Is3D == 1)
                  {
                    plan->r_strides[2] = plan->fft_dimensions[2];
                    plan->r_strides[1] = plan->fft_dimensions[1]*plan->fft_dimensions[2];
                  }

                std::cout << "IP2NIP: reset strides ..." << std::endl;
                std::cout << plan->r_strides[0] << " "
                          << plan->r_strides[1] << " "
                          << plan->r_strides[2] << " "
                          << plan->r_strides[3] << std::endl;

                *reset_r_strides = 1;
              }

            *reset_placement = 1;
            plan->placement = 0;
          }
      }

    return;
}


hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    int64_t reset_r_strides = 0;
    int64_t reset_placement = 0;

    std::cout << "hipfftExecR2C checking placement ..." << std::endl;
    CheckPlacement(plan, (void*)idata, (void*)odata, &reset_placement, &reset_r_strides);

    int64_t new_strides[4];
    if (reset_r_strides == 1)
      {
        for (int i = 0; i < 4; i++) new_strides[i] = plan->r_strides[i];
      }

    H4I::MKLShim::fftExecR2C(plan->ctxt, plan->descSR, idata, (float _Complex *)odata,
                             reset_placement, reset_r_strides, new_strides);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    int64_t reset_r_strides = 0;
    int64_t reset_placement = 0;

    std::cout << "hipfftExecC2R checking placement ..." << std::endl;
    CheckPlacement(plan, (void*)idata, (void*)odata, &reset_placement, &reset_r_strides);

    int64_t new_strides[4];
    if (reset_r_strides == 1)
      {
        for (int i = 0; i < 4; i++) new_strides[i] = plan->r_strides[i];
      }

    H4I::MKLShim::fftExecC2R(plan->ctxt, plan->descSR, (float _Complex *)idata, odata,
                             reset_placement, reset_r_strides, new_strides);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2C(hipfftHandle plan,
                           hipfftComplex* idata,
                           hipfftComplex* odata,
                           int direction)
{
    hipfftResult result = HIPFFT_SUCCESS;

    switch (direction)
    {
        case HIPFFT_FORWARD:
        {
            H4I::MKLShim::fftExecC2Cforward(plan->ctxt, plan->descSC, (float _Complex *)idata, (float _Complex *)odata); 
            break;
        }
        case HIPFFT_BACKWARD:
        {
            H4I::MKLShim::fftExecC2Cbackward(plan->ctxt, plan->descSC, (float _Complex *)idata, (float _Complex *)odata);
            break;
        }
	default:
        {
            result = HIPFFT_INVALID_VALUE;
	    return result;
        }
    }

    return result;
};

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{
    int64_t reset_r_strides = 0;
    int64_t reset_placement = 0;

    std::cout << "hipfftExecD2Z checking placement ..." << std::endl;
    CheckPlacement(plan, (void*)idata, (void*)odata, &reset_placement, &reset_r_strides);

    int64_t new_strides[4];
    if (reset_r_strides == 1)
      {
        for (int i = 0; i < 4; i++) new_strides[i] = plan->r_strides[i];
      }

    H4I::MKLShim::fftExecD2Z(plan->ctxt, plan->descDR, idata, (double _Complex *)odata,
                             reset_placement, reset_r_strides, new_strides);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{
    int64_t reset_r_strides = 0;
    int64_t reset_placement = 0;

    std::cout << "hipfftExecC2R checking placement ..." << std::endl;
    CheckPlacement(plan, (void*)idata, (void*)odata, &reset_placement, &reset_r_strides);

    int64_t new_strides[4];
    if (reset_r_strides == 1)
      {
        for (int i = 0; i < 4; i++) new_strides[i] = plan->r_strides[i];
      }

    H4I::MKLShim::fftExecZ2D(plan->ctxt, plan->descDR, (double _Complex *)idata, odata,
                             reset_placement, reset_r_strides, new_strides);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2Z(hipfftHandle plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int direction)
{
    hipfftResult result = HIPFFT_SUCCESS;

    switch (direction)
    {
        case HIPFFT_FORWARD:
        {
            H4I::MKLShim::fftExecZ2Zforward(plan->ctxt, plan->descDC, (double _Complex *)idata, (double _Complex *)odata);
            break;
        }
        case HIPFFT_BACKWARD:
        {
            H4I::MKLShim::fftExecZ2Zbackward(plan->ctxt, plan->descDC, (double _Complex *)idata, (double _Complex *)odata);
            break;
        }
        default:
        {
            result = HIPFFT_INVALID_VALUE;
            return result;
        }
    }

    return result;
};

