
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"
#include "spectrum_analyzer.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, float* idata, float* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_c2c_fwd;
    hipfftHandle plan_c2c_bwd;

    int not_in_place = 1;
    int nx = 16; 
    int ny = 18;
    int nz = 20;
    int N_alloc;
    int i,j,k;
   
    hipfftType typ = HIPFFT_C2C;
    hipfftResult result;
    hipError_t hip_error;

    // make the forward plan
    result = hipfftPlan3d(&plan_c2c_fwd, nz, ny, nx, HIPFFT_C2C);

    result = hipfftPlan3d(&plan_c2c_bwd, nz, ny, nx, HIPFFT_C2C);
    
    std::cout << std::scientific << std::setprecision(8);

    // local memory
    std::complex<float> *cx = NULL;
    std::complex<float> *cy = NULL;

    N_alloc = nx*ny*nz;

    int Nbytes = N_alloc*sizeof(std::complex<float>);
    cx = (std::complex<float>*)malloc(Nbytes);
    cy = (std::complex<float>*)malloc(Nbytes);

    // initialize
    float anx = float(nx);
    float any = float(ny);
    float anz = float(nz);
    float dx = 2.0*M_PI/(anx);    
    float dy = 2.0*M_PI/(any);
    float dz = 2.0*M_PI/(anz);
    float value;
    int offset;

    // use whole freqencies for the spectrum analysis so that
    // we can capture the initialized signal
    // NOTE: freq <= {nx/2,ny/2,nz/2} to be captured by FFT
    float freq[3] = {1.0,2.0,3.0};

    // amplitude will be returned in the spectrum analysis at the 
    // assigned freqency (if frequencies are whole numbers)
    float amp[3] = {1.0,0.75,0.5};

    // initialize the field in physical space
    for (k = 0; k < nz; k++)
      {
        for (j = 0; j < ny; j++)
          {
            for (i = 0; i < nx; i++)
              {
                offset = k*ny*nx + j*nx + i;
                value = amp[0]*sin(freq[0]*i*dx)
                      + amp[1]*sin(freq[1]*j*dy)
                      + amp[2]*sin(freq[2]*k*dz);
                cx[offset] = std::complex<float>(value,0.0);
                cy[offset] = std::complex<float>(0.0,0.0);
              }
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    std::complex<float> *x = NULL;
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

    // need a complex array on the CPU for the spectrum analysis
    std::complex<float> *sx;
    sx = (std::complex<float>*)malloc(Nbytes);

    // Copy result back to host
    if (not_in_place)
      {
        hip_error = hipMemcpy(sx, y, Nbytes, hipMemcpyDeviceToHost);
      }
    else
      {
        hip_error = hipMemcpy(sx, x, Nbytes, hipMemcpyDeviceToHost);
      }

    // single function for all dimensions ... handles the 
    // the contraction of the leading index 
    int IsR2C = 0;
    Get_Frequency_Spectrum(IsR2C, nx, ny, nz, sx);

    // cleanup
    free(sx);

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
    double factor = anx*any*anz;

    for (k = 0; k < nz; k++)
      {
        for (j = 0; j < ny; j++)
          {
            for (i = 0; i < nx; i++)
              {
                offset = k*ny*nx + j*nx + i;

                // rescale cy to account for the scaling applied during the fft transforms
                cy[offset] /= factor;

                local_error = fabs(cx[offset] - cy[offset]);
                if (local_error > max_error) max_error = local_error;
              }
          }
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

#if 0
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
#endif

    std::cout << "FINISHED!" << std::endl;

    return(0);
}
