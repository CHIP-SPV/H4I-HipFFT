
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfftHandle.h"
#include "hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, float* idata, float* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_c2c_fwd;
    hipfftHandle plan_c2c_bwd;

    int not_in_place = 0;
    int nx = 16; 
    int ny = 16;
    int N_alloc;
    int Nbytes;

    int batch;
   
    hipfftType typ = HIPFFT_C2C;
    hipfftResult result;
    hipError_t hip_error;

    batch = ny;

    // make the forward plan
    result = hipfftPlanMany(&plan_c2c_fwd, 1, &nx,
                            &nx,1,nx,
                            &nx,1,nx,
                            HIPFFT_C2C, batch);

    result = hipfftPlanMany(&plan_c2c_bwd, 1, &nx,
                            &nx,1,nx,
                            &nx,1,nx,
                            HIPFFT_C2C, batch);

    std::cout << std::scientific << std::setprecision(8);

    std::cout << "check plan_c2c_fwd" << std::endl;
    checkPlan(plan_c2c_fwd);
    std::cout << "check plan_c2c_bwd" << std::endl;
    checkPlan(plan_c2c_bwd);

    // local memory
    std::complex<float> *cx = NULL;
    std::complex<float> *cy = NULL;

    N_alloc = nx*ny;

    Nbytes = N_alloc*sizeof(std::complex<float>);
    cx = (std::complex<float>*)malloc(Nbytes);
    cy = (std::complex<float>*)malloc(Nbytes);

    // initialize
    int offset;
    float anx = float(nx);
    float dx = 2.0*M_PI/(anx);    
    for (int j = 0; j < ny; j++)
      {
        for (int i = 0; i < nx; i++)
          {
            offset = j*nx + i;

            cx[offset] = std::complex<float>(sin(i*dx),0.0);
            cy[offset] = std::complex<float>(0.0,0.0);
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    float *x = NULL;
    hip_error = hipMalloc(&x, (Nbytes));
    hip_error = hipMemcpy(x, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    std::complex<float>*y = NULL;
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
	result = hipfftExecC2C(plan_c2c_fwd, (hipfftComplex*)x, (hipfftComplex*)y, HIPFFT_FORWARD);
      }
    else
      {
	result = hipfftExecC2C(plan_c2c_fwd, (hipfftComplex*)x, (hipfftComplex*)x, HIPFFT_FORWARD);
      }
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecC2C(plan_c2c_bwd, (hipfftComplex*)y, (hipfftComplex*)x, HIPFFT_BACKWARD);
      }
    else
      {
	result = hipfftExecC2C(plan_c2c_bwd, (hipfftComplex*)x, (hipfftComplex*)x, HIPFFT_BACKWARD);
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
    result = hipfftDestroy(plan_c2c_fwd);
    result = hipfftDestroy(plan_c2c_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // we should be able to reuse the plan handle now
    result = hipfftCreate(&plan_c2c_fwd);
    result = hipfftCreate(&plan_c2c_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // destroy the plan
    result = hipfftDestroy(plan_c2c_fwd);
    result = hipfftDestroy(plan_c2c_bwd);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
