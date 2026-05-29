#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include <vector>
#include "hipfftHandle.h"
#include "hipfft.h"
#include "hipfft-version.h"
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "h4i/mklshim/mklshim.h"
#include "h4i/mklshim/onemklfft.h"


hipfftResult hipfftCreate(hipfftHandle *plan)
{
    // create the plan handle
    hipfftHandle h = new hipfftHandle_t;

    int nHandles;
    hipGetBackendNativeHandles((uintptr_t)0, 0, &nHandles);

    // Replace VLA with std::vector
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)NULL, handles.data(), 0);
    // MKLShim::Create expects handles[0] = backendName, handles[1..n-1] = native handles
    auto *ctxt = H4I::MKLShim::Create(handles.data(), nHandles);

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


hipfftResult hipfftPlan3d(hipfftHandle *plan, int nx, int ny, int nz, hipfftType type)
{
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        *plan = nullptr;
        return HIPFFT_INVALID_VALUE;
    }
    hipfftHandle h = nullptr;
    hipfftResult result = hipfftCreate(&h);
    if (result != HIPFFT_SUCCESS) return result;
    result = hipfftMakePlan3d(h, nx, ny, nz, type, nullptr);
    *plan = h;
    return result;
}


hipfftResult hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t *workSize)
{
    std::vector<std::int64_t> dimensions {(int64_t)nx, (int64_t)ny, (int64_t)nz};
    switch (type) {
        case HIPFFT_R2C:
        case HIPFFT_C2R:
            plan->descSR = H4I::MKLShim::createFFTDescriptorSR(plan->ctxt, dimensions);
            break;
        case HIPFFT_D2Z:
        case HIPFFT_Z2D:
            plan->descDR = H4I::MKLShim::createFFTDescriptorDR(plan->ctxt, dimensions);
            break;
        case HIPFFT_C2C:
            plan->descSC = H4I::MKLShim::createFFTDescriptorSC(plan->ctxt, dimensions);
            break;
        case HIPFFT_Z2Z:
            plan->descDC = H4I::MKLShim::createFFTDescriptorDC(plan->ctxt, dimensions);
            break;
        default:
            return HIPFFT_INVALID_PLAN;
    }
    return HIPFFT_SUCCESS;
}


hipfftResult hipfftExecZ2Z(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleComplex *odata, int direction)
{
    switch (direction) {
        case HIPFFT_FORWARD:
            H4I::MKLShim::fftExecZ2Zforward(plan->ctxt, plan->descDC,
                (double _Complex *)idata, (double _Complex *)odata);
            return HIPFFT_SUCCESS;
        case HIPFFT_BACKWARD:
            H4I::MKLShim::fftExecZ2Zbackward(plan->ctxt, plan->descDC,
                (double _Complex *)idata, (double _Complex *)odata);
            return HIPFFT_SUCCESS;
        default:
            return HIPFFT_INVALID_VALUE;
    }
}


hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal *idata, hipfftDoubleComplex *odata)
{
    H4I::MKLShim::fftExecD2Z(plan->ctxt, plan->descDR, idata, (double _Complex *)odata);
    return HIPFFT_SUCCESS;
}


hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex *idata, hipfftDoubleReal *odata)
{
    H4I::MKLShim::fftExecZ2D(plan->ctxt, plan->descDR, (double _Complex *)idata, odata);
    return HIPFFT_SUCCESS;
}


hipfftResult hipfftDestroy(hipfftHandle plan)
{
    if (plan == nullptr) return HIPFFT_INVALID_PLAN;
    if (plan->descSR != nullptr) H4I::MKLShim::destroyFFTDescriptorSR(plan->ctxt, plan->descSR);
    if (plan->descSC != nullptr) H4I::MKLShim::destroyFFTDescriptorSC(plan->ctxt, plan->descSC);
    if (plan->descDR != nullptr) H4I::MKLShim::destroyFFTDescriptorDR(plan->ctxt, plan->descDR);
    if (plan->descDC != nullptr) H4I::MKLShim::destroyFFTDescriptorDC(plan->ctxt, plan->descDC);
    delete plan;
    return HIPFFT_SUCCESS;
}


// chipStar dispatches kernels on its own queue; the MKLShim context already
// holds the SYCL queue used for the FFT.  hipFFT consumers (e.g. OpenMM) call
// hipfftSetStream to associate a hipStream_t with the plan — accept it as a
// no-op so existing code paths link and run; once MKLShim exposes a
// stream-binding API this can be wired through.
hipfftResult hipfftSetStream(hipfftHandle /*plan*/, hipStream_t /*stream*/)
{
    return HIPFFT_SUCCESS;
}


hipfftResult hipfftGetVersion(int *version)
{
    if (version == nullptr) return HIPFFT_INVALID_VALUE;
    *version = hipfftVersionMajor * 10000 + hipfftVersionMinor * 100 + hipfftVersionPatch;
    return HIPFFT_SUCCESS;
}
