
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft/hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, float* idata, float* odata);

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

    // make the forward plan
    result = hipfftPlan1d(&plan_r2c, nx, HIPFFT_R2C, 1);

    result = hipfftPlan1d(&plan_c2r, nx, HIPFFT_C2R, 1);
    
    std::cout << std::scientific << std::setprecision(8);
    // std::cout << " plan_r2c scale factor = " << plan_r2c->scale_factor << std::endl;
    // std::cout << " plan_c2r scale factor = " << plan_c2r->scale_factor << std::endl;

    // checkPlan(plan_r2c);
    // checkPlan(plan_c2r);

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
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
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

    // testPlan(plan_r2c, Nbytes, (hipfftReal*)x, (hipfftReal*)cy);

    // Copy result back to host
    hip_error = hipMemcpy(cy, x, Nbytes, hipMemcpyDeviceToHost);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    //  error check
    double local_error = 0.0;
    double max_error = 0.0;
    for (int i = 0; i < nx; i++)
      {
        // rescale cy to account for the scaling applied during the fft transforms
        cy[i] /= anx;

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

    // destroy the plan
    result = hipfftDestroy(plan_r2c);
    result = hipfftDestroy(plan_c2r);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

#if 0
    // we should be able to reuse the plan handle now
    result = hipfftCreate(&plan_r2c);
    result = hipfftCreate(&plan_c2r);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // destroy the plan
    result = hipfftDestroy(plan_r2c);
    result = hipfftDestroy(plan_c2r);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();
#endif

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
