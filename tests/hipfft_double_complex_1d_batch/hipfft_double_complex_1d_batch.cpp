
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, double* idata, double* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_z2z_fwd;
    hipfftHandle plan_z2z_bwd;

    int not_in_place = 0;
    int nx = 16; 
    int ny = 16;
    int N_alloc;
    int Nbytes;

    int batch;
   
    hipfftType typ = HIPFFT_Z2Z;
    hipfftResult result;
    hipError_t hip_error;

    batch = ny;

    // make the forward plan
    result = hipfftPlanMany(&plan_z2z_fwd, 1, &nx,
                            &nx,1,nx,
                            &nx,1,nx,
                            HIPFFT_Z2Z, batch);

    result = hipfftPlanMany(&plan_z2z_bwd, 1, &nx,
                            &nx,1,nx,
                            &nx,1,nx,
                            HIPFFT_Z2Z, batch);

    std::cout << std::scientific << std::setprecision(8);

    std::cout << "check plan_z2z_fwd" << std::endl;
    checkPlan(plan_z2z_fwd);
    std::cout << "check plan_z2z_bwd" << std::endl;
    checkPlan(plan_z2z_bwd);

    // local memory
    std::complex<double> *cx = NULL;
    std::complex<double> *cy = NULL;

    N_alloc = nx*ny;

    Nbytes = N_alloc*sizeof(std::complex<double>);
    cx = (std::complex<double>*)malloc(Nbytes);
    cy = (std::complex<double>*)malloc(Nbytes);

    // initialize
    int offset;
    double anx = double(nx);
    double dx = 2.0*M_PI/(anx);    
    for (int j = 0; j < ny; j++)
      {
        for (int i = 0; i < nx; i++)
          {
            offset = j*nx + i;

            cx[offset] = std::complex<double>(sin(i*dx),0.0);
            cy[offset] = std::complex<double>(0.0,0.0);
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    double *x = NULL;
    hip_error = hipMalloc(&x, (Nbytes));
    hip_error = hipMemcpy(x, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    std::complex<double>*y = NULL;
    if (not_in_place)
      {
	hip_error = hipMalloc(&y, Nbytes);
      }

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the forward transform
    if (not_in_place)
      {
	result = hipfftExecZ2Z(plan_z2z_fwd, (hipfftDoubleComplex*)x, (hipfftDoubleComplex*)y, HIPFFT_FORWARD);
      }
    else
      {
	result = hipfftExecZ2Z(plan_z2z_fwd, (hipfftDoubleComplex*)x, (hipfftDoubleComplex*)x, HIPFFT_FORWARD);
      }
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecZ2Z(plan_z2z_bwd, (hipfftDoubleComplex*)y, (hipfftDoubleComplex*)x, HIPFFT_BACKWARD);
      }
    else
      {
	result = hipfftExecZ2Z(plan_z2z_bwd, (hipfftDoubleComplex*)x, (hipfftDoubleComplex*)x, HIPFFT_BACKWARD);
      }

#endif
    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // Copy result back to host
    hip_error = hipMemcpy(cy, x, Nbytes, hipMemcpyDeviceToHost);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    //  error check
    double local_error = 0.0;
    double max_error = 0.0;
    for (int j = 0; j < batch; j++)
      {
        for (int i = 0; i < nx; i++)
          {
            offset = j*nx + i;

            // rescale cy to account for the scaling applied during the fft transforms
            cy[offset] /= anx;

            local_error = fabs(cx[offset] - cy[offset]);
            if (local_error > max_error) max_error = local_error;

            // std::cout << cx[offset] << " " << cy[offset] << " " << (cx[offset] - cy[offset]) << std::endl;
          }
        // std::cout << std::endl;
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
    result = hipfftDestroy(plan_z2z_fwd);
    result = hipfftDestroy(plan_z2z_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

#if 0
    // we should be able to reuse the plan handle now
    result = hipfftCreate(&plan_z2z_fwd);
    result = hipfftCreate(&plan_z2z_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // destroy the plan
    result = hipfftDestroy(plan_z2z_fwd);
    result = hipfftDestroy(plan_z2z_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();
#endif

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
