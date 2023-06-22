
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
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
        result = HIPFFT_INVALID_PLAN;
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
    H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descSR);
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
                          int nx,
                          int ny,
                          hipfftType type)
{
    hipfftResult result = HIPFFT_SUCCESS;

#if 0
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
#endif

    return result;
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    // std::cout << "In ExecR2C " << std::endl;
    H4I::MKLShim::fftExecR2C(plan->ctxt, plan->descSR, idata, (float _Complex *)odata);
    // std::cout << "leaving hipfftExecR2C " << std::endl;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    // std::cout << "In ExecC2R " << std::endl;
    H4I::MKLShim::fftExecC2R(plan->ctxt, plan->descSR, (float _Complex *)idata, odata);
    return HIPFFT_SUCCESS;
}

#if 0
hipfftResult hipfftExecC2C(hipfftHandle plan,
                           hipfftComplex* idata,
                           hipfftComplex* odata,
                           int direction)
{
    std::cout << "In ExecC2C " << std::endl;

    int _direction;
    switch (direction)
    {
        case HIPFFT_FORWARD:
        {
            _direction = 0; 
            break;
        }
        case HIPFFT_BACKWARD:
        {
            _direction = 1;
            break;
        }
    }

    // H4I::MKLShim::fftExecC2C(plan->ctxt, plan->descSC, (float _Complex *)idata, (float _Complex *)odata, _direction);
    return HIPFFT_SUCCESS;
};
#endif

