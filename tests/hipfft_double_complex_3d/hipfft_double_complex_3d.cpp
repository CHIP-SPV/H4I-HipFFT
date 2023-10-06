
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfftHandle.h"
#include "hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"
#include "spectrum_analyzer.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, double* idata, double* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_z2z_fwd;
    hipfftHandle plan_z2z_bwd;

    int not_in_place = 0;
    int nx = 16; 
    int ny = 18;
    int nz = 20;
    int N_alloc;
    int i,j,k;
   
    hipfftType typ = HIPFFT_Z2Z;
    hipfftResult result;
    hipError_t hip_error;

    // make the forward plan
    result = hipfftPlan3d(&plan_z2z_fwd, nz, ny, nx, HIPFFT_Z2Z);

    result = hipfftPlan3d(&plan_z2z_bwd, nz, ny, nx, HIPFFT_Z2Z);
    
    std::cout << std::scientific << std::setprecision(8);

    // local memory
    std::complex<double> *cx = NULL;
    std::complex<double> *cy = NULL;

    N_alloc = nx*ny*nz;

    int Nbytes = N_alloc*sizeof(std::complex<double>);
    cx = (std::complex<double>*)malloc(Nbytes);
    cy = (std::complex<double>*)malloc(Nbytes);

    // initialize
    double anx = double(nx);
    double any = double(ny);
    double anz = double(nz);
    double dx = 2.0*M_PI/(anx);    
    double dy = 2.0*M_PI/(any);
    double dz = 2.0*M_PI/(anz);
    double value;
    int offset;

    // use whole freqencies for the spectrum analysis so that
    // we can capture the initialized signal
    // NOTE: freq <= {nx/2,ny/2,nz/2} to be captured by FFT
    double freq[3] = {1.0,2.0,3.0};

    // amplitude will be returned in the spectrum analysis at the 
    // assigned freqency (if frequencies are whole numbers)
    double amp[3] = {1.0,0.75,0.5};

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
                cx[offset] = std::complex<double>(value,0.0);
                cy[offset] = std::complex<double>(0.0,0.0);
              }
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    std::complex<double> *x = NULL;
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

    // need a complex array on the CPU for the spectrum analysis
    std::complex<double> *sx;
    sx = (std::complex<double>*)malloc(Nbytes);

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
    int IsD2Z = 0;
    Get_Frequency_Spectrum(IsD2Z, nx, ny, nz, sx);

    // cleanup
    free(sx);

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
