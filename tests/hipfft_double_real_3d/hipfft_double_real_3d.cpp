
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft/hipfft.h"
// #include "hip/hip_runtime.h"
// #include "hip/hip_interop.h"
#include "spectrum_analyzer.h"

void checkPlan(hipfftHandle plan);
void testPlan(hipfftHandle plan, int Nbytes, double* idata, double* odata);

int main(int argc, char **argv)
{
    hipfftHandle plan_d2z;
    hipfftHandle plan_z2d;

    int not_in_place = 1;
    int nx = 16; 
    int ny = 18;
    int nz = 20;
    int p_nx;
    int w_nx;
    int N_alloc;
    int i,j,k;
   
    hipfftType typ = HIPFFT_D2Z;
    hipfftResult result;
    hipError_t hip_error;

    // make the forward plan
    result = hipfftPlan3d(&plan_d2z, nz, ny, nx, HIPFFT_D2Z);

    result = hipfftPlan3d(&plan_z2d, nz, ny, nx, HIPFFT_Z2D);
    
    std::cout << std::scientific << std::setprecision(8);
    // std::cout << " plan_d2z scale factor = " << plan_d2z->scale_factor << std::endl;
    // std::cout << " plan_z2d scale factor = " << plan_z2d->scale_factor << std::endl;

    // checkPlan(plan_d2z);
    // checkPlan(plan_z2d);


    // local memory
    double *cx = NULL;
    double *cy = NULL;

    // deal with the D2Z contraction from physical space to wavenumber space
    w_nx = (nx/2) + 1;

    if (not_in_place)
      {
        p_nx = nx;
      }
    else // in-place transform
      {
        p_nx = 2*w_nx;
      }

    N_alloc = p_nx*ny*nz;

    int Nbytes = N_alloc*sizeof(double);
    cx = (double*)malloc(Nbytes);
    cy = (double*)malloc(Nbytes);

    // initialize
    double anx = double(nx);
    double any = double(ny);
    double anz = double(nz);
    double dx = 2.0*M_PI/(anx);    
    double dy = 2.0*M_PI/(any);
    double dz = 2.0*M_PI/(anz);
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
                offset = k*ny*p_nx + j*p_nx + i;
                cx[offset] = amp[0]*sin(freq[0]*i*dx) 
                           + amp[1]*sin(freq[1]*j*dy)
                           + amp[2]*sin(freq[2]*k*dz);
                cy[offset] = 0.0;
              }
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    double *x = NULL;
    hip_error = hipMalloc(&x, (Nbytes));
    hip_error = hipMemcpy(x, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    N_alloc = w_nx*ny*nz;
    int wNbytes = N_alloc*sizeof(std::complex<double>);
    std::complex<double>*y = NULL;
    if (not_in_place)
      {
	hip_error = hipMalloc(&y, wNbytes);
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

    // need a complex array on the CPU for the spectrum analysis
    std::complex<double> *sx;
    sx = (std::complex<double>*)malloc(wNbytes);

    // Copy result back to host
    if (not_in_place)
      {
        hip_error = hipMemcpy(sx, y, wNbytes, hipMemcpyDeviceToHost);
      }
    else
      {
        hip_error = hipMemcpy(sx, x, wNbytes, hipMemcpyDeviceToHost);
      }

    // single function for all dimensions ... handles the 
    // the contraction of the leading index 
    int IsD2Z = 1;
    Get_Frequency_Spectrum(IsD2Z, nx, ny, nz, sx);

    // cleanup
    free(sx);

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
                offset = k*ny*p_nx + j*p_nx + i;

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
