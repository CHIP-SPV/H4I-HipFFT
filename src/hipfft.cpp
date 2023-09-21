
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include <vector>
#include "hipfftHandle.h"
#include "hipfft.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"


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

    // create the appropriate descriptor for the fft plan
    switch (type)
    {
       case HIPFFT_R2C:
       case HIPFFT_C2R:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);
          plan->descSR = desc;
          break;
       }
       case HIPFFT_D2Z:
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

    // define the dimensions vector
    // NOTE: may need to swap these ... need to look at the intel docs, 
    //       but I think intel follows the same convention as hipfft
    std::vector<std::int64_t> dimensions {(int64_t)nx, (int64_t)ny, (int64_t)nz};

    // create the appropriate descriptor for the fft plan
    switch (type)
    {
       case HIPFFT_R2C:
       case HIPFFT_C2R:
       {
          auto *desc = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt,dimensions);
          plan->descSR = desc;
          break;
       }
       case HIPFFT_D2Z:
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

          // set parameters
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

          H4I::MKLShim::setFFTPlanValuesSC(plan->ctxt, desc,
                                           (int64_t)istride, (int64_t)idist,
                                           (int64_t)ostride, (int64_t)odist,
                                           (int64_t)batch);

          // set parameters for intel onemkl fft
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

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    H4I::MKLShim::fftExecR2C(plan->ctxt, plan->descSR, idata, (float _Complex *)odata);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    H4I::MKLShim::fftExecC2R(plan->ctxt, plan->descSR, (float _Complex *)idata, odata);
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
    H4I::MKLShim::fftExecD2Z(plan->ctxt, plan->descDR, idata, (double _Complex *)odata);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{
    H4I::MKLShim::fftExecZ2D(plan->ctxt, plan->descDR, (double _Complex *)idata, odata);
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

