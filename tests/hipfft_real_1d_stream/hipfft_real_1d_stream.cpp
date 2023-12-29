
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft/hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"

int main(int argc, char **argv)
{
    hipfftHandle plan_r2c;
    hipfftHandle plan_c2r;

    int not_in_place = 1;
    int nx = 16; 
    int ny = 16;
    int N_alloc;
   
    hipfftType typ = HIPFFT_R2C;
    hipfftResult result;
    hipError_t hip_error;

    hipStream_t stream;

    hip_error = hipStreamCreate(&stream);

    // make the forward plan
    result = hipfftPlan1d(&plan_r2c, nx, HIPFFT_R2C, 1);
    result = hipfftPlan1d(&plan_c2r, nx, HIPFFT_C2R, 1);

    result = hipfftSetStream(plan_r2c, stream);
    result = hipfftSetStream(plan_c2r, stream);
    
    std::cout << std::scientific << std::setprecision(8);

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

    float factor = 1.0;

    // Wait for events assigned to the stream to finish;
    hip_error = hipStreamSynchronize(stream);

    // compute the forward transform
    if (not_in_place)
      {
        result = hipfftExecR2C(plan_r2c, (hipfftReal*)x, (hipfftComplex*)y);
      }
    else
      {
        result = hipfftExecR2C(plan_r2c, (hipfftReal*)x, (hipfftComplex*)x);
      }

    // Wait for events assigned to the stream to finish;
    hip_error = hipStreamSynchronize(stream);

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

    // Wait for events assigned to the stream to finish;
    hip_error = hipStreamSynchronize(stream);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // accumulate the scaling factor
    factor *= anx;

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
        cy[i] /= factor;

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

    // make sure the stream is finished
    hip_error = hipStreamSynchronize(stream);
    hip_error = hipStreamDestroy(stream);
        
    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
