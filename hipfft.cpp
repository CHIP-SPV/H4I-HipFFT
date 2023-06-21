
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


void say_hello()
{
    std::cout << "Hello, from h4i-hipfft!\n";
}



hipfftResult hipfftCreate(hipfftHandle *plan)
{
    // create the plan handle
    hipfftHandle h = new hipfftHandle_t;

    // Next, create the context struct
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
        result = hipfftMakePlan1d(h, nx, type, batch, NULL);

        // set plan
        *plan = h;
    }
    else
    {
        *plan = NULL;
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




hipfftResult checkPlan(hipfftHandle plan)
{
    hipfftResult result = HIPFFT_SUCCESS;

    H4I::MKLShim::checkFFTQueue(plan->ctxt);

    H4I::MKLShim::checkFFTPlan(plan->ctxt, plan->descSR);

    return result;
}

hipfftResult hipfftPlan2d(hipfftHandle *plan,
                          int nx,
                          int ny,
                          hipfftType type)
{
    hipfftResult result = HIPFFT_SUCCESS;

#if 0

#endif

  return result;

}


HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal *idata, hipfftComplex *odata)
{
    std::cout << "In ExecR2C " << std::endl;
    // return plan->hipfftExecR2C_impl(idata, odata);
    return HIPFFT_SUCCESS;
}

HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex *idata, hipfftReal *odata)
{
    std::cout << "In ExecC2R " << std::endl;
    // return plan->hipfftExecC2R_impl(idata, odata);
    return HIPFFT_SUCCESS;
}

HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata,
                                         int            direction)
{
    std::cout << "In ExecC2C " << std::endl;
    // return plan->hipfftExecC2C_impl(idata, odata, direction);
    return HIPFFT_SUCCESS;
};

int main(int argc, char **argv)
{
    hipfftHandle plan_r2c;
    hipfftHandle plan_c2r;

    int not_in_place = 0;
    int nx = 16; 
    int ny = 16;
    int N_alloc;
   
    hipfftType typ = HIPFFT_R2C;
    hipfftResult result;
    hipError_t hip_error;

    say_hello();

    // make the forward plan
    result = hipfftPlan1d(&plan_r2c, nx, HIPFFT_R2C, 1);
    // result = hipfftPlan2d(&plan_r2c, nx, ny, HIPFFT_R2C);

    result = hipfftPlan1d(&plan_c2r, nx, HIPFFT_C2R, 1);
    // result = hipfftPlan2d(&plan_c2r, nx, ny, HIPFFT_C2R);
    
    std::cout << std::scientific << std::setprecision(8);
    std::cout << " plan_r2c scale factor = " << plan_r2c->scale_factor << std::endl;
    std::cout << " plan_c2r scale factor = " << plan_c2r->scale_factor << std::endl;

    result = checkPlan(plan_r2c);
    result = checkPlan(plan_c2r);

    // local memory
    float *cx = NULL;
    float *cy = NULL;

    if (not_in_place)
      {
        N_alloc = nx;
      }
    else // in-place transform
      {
        N_alloc = 2*((nx/2) + 1);
      }

    int Nbytes = N_alloc*sizeof(float);
    cx = (float*)malloc(Nbytes);
    cy = (float*)malloc(Nbytes);

    // initialize
    float anx = float(nx);
    float dx = 2.0*M_PI/(anx);    
    for (int i = 0; i < nx; i++)
      {
        cx[i] = sin(i*dx);
        cy[i] = 0.0;
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    float *x = NULL;
    hip_error = hipMalloc(&x, (Nbytes));
    hip_error = hipMemcpy(x, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    int ynx = (nx/2) + 1;
    int yNbytes = ynx*sizeof(std::complex<float>);
    std::complex<float>*y = NULL;
    if (not_in_place)
      {
	hip_error = hipMalloc(&y, yNbytes);
      }

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the forward transform
    if (not_in_place)
      {
	result = hipfftExecR2C(plan_r2c, (hipfftReal*)x, (hipfftComplex*)y);
      }
    else
      {
	result = hipfftExecR2C(plan_r2c, (hipfftReal*)x, (hipfftComplex*)x);
      }

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecC2R(plan_c2r, (hipfftComplex*)y, (hipfftReal*)x);
      }
    else
      {
	result = hipfftExecC2R(plan_c2r, (hipfftComplex*)x, (hipfftReal*)x);
      }

#endif
    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // Copy result back to host
    hip_error = hipMemcpy(cy, x, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    //  error check
    double local_error = 0.0;
    double max_error = 0.0;
    for (int i = 0; i < nx; i++)
      {
        // rescale cy to account for the scaling applied during the fft transforms
        // cy[i] /= anx;

        local_error = fabs(cx[i] - cy[i]);
        if (local_error > max_error) max_error = local_error;
      }

    std::cout << std::scientific << std::setprecision(15);
    std::cout << "max error = " << max_error << std::endl;

    // clean-up
    free(cx);
    free(cy);

    // Free device buffer
    hip_error = hipFree(x);
    if (not_in_place)
      {
       hip_error = hipFree(y);
      }

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
