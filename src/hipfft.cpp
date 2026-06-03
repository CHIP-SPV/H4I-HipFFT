#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include <vector>
#include "hipfftHandle.h"
#include "hipfft.h"
#include "hipfft-version.h"
#include "h4i/mklshim/Stream.h"
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
    // Release the per-queue MKLShim Context held since hipfftCreate.
    // Without this the Context stays in MKLShim::context_tbl keyed by
    // the underlying L0 queue handle pointer; when chipStar later
    // recycles that handle for a freshly created hipStream_t, the
    // next hipfftCreate returns the stale Context and oneMKL commits
    // against a SYCL queue whose backing L0 queue was destroyed.
    if (plan->ctxt != nullptr) H4I::MKLShim::Destroy(plan->ctxt);
    delete plan;
    return HIPFFT_SUCCESS;
}


// Bind the plan to the SYCL queue underlying the given hipStream_t so that
// FFT submissions land on the same chipStar L0 command list as the
// surrounding HIP kernels.  Without this the plan executes on whatever
// queue was current at hipfftCreate time (typically the default stream),
// which races against kernels on a non-default pmeStream and produces
// stale FFT input/output for consumers like OpenMM PME.
hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    if (plan == nullptr) return HIPFFT_INVALID_PLAN;
    int nHandles = 0;
    hipGetBackendNativeHandles((uintptr_t)stream, 0, &nHandles);
    if (nHandles <= 0) return HIPFFT_INVALID_VALUE;
    std::vector<unsigned long> handles(nHandles);
    hipGetBackendNativeHandles((uintptr_t)stream, handles.data(), 0);
    // MKLShim::SetStream discards Update's return; call Update directly so
    // we capture the (possibly different) context pointer if the table
    // already has a matching queue.
    plan->ctxt = H4I::MKLShim::Update(plan->ctxt, handles.data(), nHandles);
    if (plan->ctxt == nullptr) return HIPFFT_INVALID_VALUE;
    // The FFT plan was committed on the previous queue at create time; rebind
    // each descriptor so future compute_* submissions land on the new queue.
    if (plan->descSR != nullptr) H4I::MKLShim::rebindFFTDescriptorSR(plan->ctxt, plan->descSR);
    if (plan->descSC != nullptr) H4I::MKLShim::rebindFFTDescriptorSC(plan->ctxt, plan->descSC);
    if (plan->descDR != nullptr) H4I::MKLShim::rebindFFTDescriptorDR(plan->ctxt, plan->descDR);
    if (plan->descDC != nullptr) H4I::MKLShim::rebindFFTDescriptorDC(plan->ctxt, plan->descDC);
    return HIPFFT_SUCCESS;
}


hipfftResult hipfftGetVersion(int *version)
{
    if (version == nullptr) return HIPFFT_INVALID_VALUE;
    *version = hipfftVersionMajor * 10000 + hipfftVersionMinor * 100 + hipfftVersionPatch;
    return HIPFFT_SUCCESS;
}
