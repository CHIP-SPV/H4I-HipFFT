
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft/hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, double* idata, double* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_d2z;
    hipfftHandle plan_z2d;

    int not_in_place = 0;
    int nx = 16; 
    int ny = 16;
    int N_alloc;
   
    hipfftType typ = HIPFFT_D2Z;
    hipfftResult result;
    hipError_t hip_error;

    // make the forward plan
    result = hipfftPlan1d(&plan_d2z, nx, HIPFFT_D2Z, 1);

    result = hipfftPlan1d(&plan_z2d, nx, HIPFFT_Z2D, 1);
    
    std::cout << std::scientific << std::setprecision(8);
    // std::cout << " plan_d2z scale factor = " << plan_d2z->scale_factor << std::endl;
    // std::cout << " plan_z2d scale factor = " << plan_z2d->scale_factor << std::endl;

    // checkPlan(plan_d2z);
    // checkPlan(plan_z2d);

    // local memory
    double *cx = NULL;
    double *cy = NULL;

    if (not_in_place)
      {
        N_alloc = nx;
      }
    else // in-place transform
      {
        N_alloc = 2*((nx/2) + 1);
      }

    int Nbytes = N_alloc*sizeof(double);
    cx = (double*)malloc(Nbytes);
    cy = (double*)malloc(Nbytes);

    // initialize
    double anx = double(nx);
    double dx = 2.0*M_PI/(anx);    
    for (int i = 0; i < nx; i++)
      {
        cx[i] = sin(i*dx);
        cy[i] = 0.0;
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    double *x = NULL;
    hip_error = hipMalloc(&x, (Nbytes));
    hip_error = hipMemcpy(x, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    int ynx = (nx/2) + 1;
    int yNbytes = ynx*sizeof(std::complex<double>);
    std::complex<double>*y = NULL;
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
	result = hipfftExecD2Z(plan_d2z, (hipfftDoubleReal*)x, (hipfftDoubleComplex*)y);
      }
    else
      {
	result = hipfftExecD2Z(plan_d2z, (hipfftDoubleReal*)x, (hipfftDoubleComplex*)x);
      }
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecZ2D(plan_z2d, (hipfftDoubleComplex*)y, (hipfftDoubleReal*)x);
      }
    else
      {
	result = hipfftExecZ2D(plan_z2d, (hipfftDoubleComplex*)x, (hipfftDoubleReal*)x);
      }

#endif
    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // testPlan(plan_d2z, Nbytes, (hipfftDoubleReal*)x, (hipfftDoubleReal*)cy);

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
    result = hipfftDestroy(plan_d2z);
    result = hipfftDestroy(plan_z2d);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

#if 0
    // we should be able to reuse the plan handle now
    result = hipfftCreate(&plan_d2z);
    result = hipfftCreate(&plan_z2d);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // destroy the plan
    result = hipfftDestroy(plan_d2z);
    result = hipfftDestroy(plan_z2d);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();
#endif

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
