
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

    int cnx = nx/2 + 1;
    int nxp;

    int batch;
    int N_alloc;
   
    hipfftType typ = HIPFFT_D2Z;
    hipfftResult result;
    hipError_t hip_error;

    if (not_in_place)
      {
        nxp = nx;
      }
    else // in-place transform
      {
        nxp = 2*cnx;
      }

    batch = ny; 

    // make the forward plan
    result = hipfftPlanMany(&plan_d2z, 1, &nx, 
                            &nxp,1,nxp,
                            &cnx,1,cnx,
                            HIPFFT_D2Z, batch);

    result = hipfftPlanMany(&plan_z2d, 1, &nx, 
                            &cnx,1,cnx,
                            &nxp,1,nxp,
                            HIPFFT_Z2D, batch);
  
    std::cout << std::scientific << std::setprecision(8);
    // std::cout << " plan_d2z scale factor = " << plan_d2z->scale_factor << std::endl;
    // std::cout << " plan_z2d scale factor = " << plan_z2d->scale_factor << std::endl;

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    // std::cout << "check plan_d2z" << std::endl;
    // checkPlan(plan_d2z);
    // std::cout << "check plan_z2d" << std::endl;
    // checkPlan(plan_z2d);

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

#if 1
    // local memory
    double *cx = NULL;
    double *cy = NULL;

    N_alloc = nxp*ny; 

    int Nbytes = N_alloc*sizeof(double);
    cx = (double*)malloc(Nbytes);
    cy = (double*)malloc(Nbytes);

    // initialize
    int offset;
    double anx = double(nx);
    double dx = 2.0*M_PI/(anx);    
    for (int j = 0; j < ny; j++)
      {
        for (int i = 0; i < nx; i++)
          {
            offset = j*nxp + i;
            cx[offset] = sin(i*dx);
            cy[offset] = 0.0;
          }
      }
    
    // device memory
    // may need to make this void to deal with the hipfftExec calls
    double *d_cx = NULL;
    hip_error = hipMalloc(&d_cx, (Nbytes));
    hip_error = hipMemcpy(d_cx, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    int yNbytes = ny*cnx*sizeof(std::complex<double>);
    std::complex<double>*d_cy = NULL;
    if (not_in_place)
      {
	hip_error = hipMalloc(&d_cy, yNbytes);
      }

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the forward transform
    if (not_in_place)
      {
	result = hipfftExecD2Z(plan_d2z, (hipfftDoubleReal*)d_cx, (hipfftDoubleComplex*)d_cy);
      }
    else
      {
	result = hipfftExecD2Z(plan_d2z, (hipfftDoubleReal*)d_cx, (hipfftDoubleComplex*)d_cx);
      }
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecZ2D(plan_z2d, (hipfftDoubleComplex*)d_cy, (hipfftDoubleReal*)d_cx);
      }
    else
      {
	result = hipfftExecZ2D(plan_z2d, (hipfftDoubleComplex*)d_cx, (hipfftDoubleReal*)d_cx);
      }

#endif
    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // testPlan(plan_d2z, Nbytes, (hipfftDoubleReal*)d_cx, (hipfftDoubleReal*)cy);

    // Copy result back to host
    hip_error = hipMemcpy(cy, d_cx, Nbytes, hipMemcpyDeviceToHost);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    //  error check
    double local_error = 0.0;
    double max_error = 0.0;
    for (int j = 0; j < batch; j++)
      {
        for (int i = 0; i < nx; i++)
          {
            offset = j*nxp + i;

            // rescale cy to account for the scaling applied during the fft transforms
            cy[offset] /= anx;

            local_error = fabs(cx[offset] - cy[offset]);
            if (local_error > max_error) max_error = local_error;
          }
      }

    std::cout << std::scientific << std::setprecision(15);
    std::cout << "max error = " << max_error << std::endl;

    // clean-up
    free(cx);
    free(cy);

    // Free device buffer
    hip_error = hipFree(d_cx);
    if (not_in_place)
      {
       hip_error = hipFree(d_cy);
      }

#endif

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
