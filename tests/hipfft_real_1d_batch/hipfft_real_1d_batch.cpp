
#include <iostream>
#include <iomanip>
#include <fstream>
#include "math.h"
#include <complex>
#include "hipfft.h"
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

    int cnx = nx/2 + 1;
    int nxp;

    int batch;
    int N_alloc;
   
    hipfftType typ = HIPFFT_R2C;
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
    result = hipfftPlanMany(&plan_r2c, 1, &nx, 
                            &nxp,1,nxp,
                            &cnx,1,cnx,
                            HIPFFT_R2C, batch);
#if 1
    result = hipfftPlanMany(&plan_c2r, 1, &nx, 
                            &cnx,1,cnx,
                            &nxp,1,nxp,
                            HIPFFT_C2R, batch);
#endif    
    std::cout << std::scientific << std::setprecision(8);
    // std::cout << " plan_r2c scale factor = " << plan_r2c->scale_factor << std::endl;
    // std::cout << " plan_c2r scale factor = " << plan_c2r->scale_factor << std::endl;

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

    std::cout << "check plan_r2c" << std::endl;
    checkPlan(plan_r2c);
#if 1
    std::cout << "check plan_c2r" << std::endl;
    checkPlan(plan_c2r);
#endif

    // make sure Q is finished
    hip_error = hipDeviceSynchronize();

#if 1
    // local memory
    float *cx = NULL;
    float *cy = NULL;

    N_alloc = nxp*ny; 

    int Nbytes = N_alloc*sizeof(float);
    cx = (float*)malloc(Nbytes);
    cy = (float*)malloc(Nbytes);

    // initialize
    int offset;
    float anx = float(nx);
    float dx = 2.0*M_PI/(anx);    
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
    float *d_cx = NULL;
    hip_error = hipMalloc(&d_cx, (Nbytes));
    hip_error = hipMemcpy(d_cx, cx, Nbytes, hipMemcpyHostToDevice);

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    int yNbytes = ny*cnx*sizeof(std::complex<float>);
    std::complex<float>*d_cy = NULL;
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
	result = hipfftExecR2C(plan_r2c, (hipfftReal*)d_cx, (hipfftComplex*)d_cy);
      }
    else
      {
	result = hipfftExecR2C(plan_r2c, (hipfftReal*)d_cx, (hipfftComplex*)d_cx);
      }
#endif

    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

#if 1
    // compute the inverse transform
    if (not_in_place)
      {
	result = hipfftExecC2R(plan_c2r, (hipfftComplex*)d_cy, (hipfftReal*)d_cx);
      }
    else
      {
	result = hipfftExecC2R(plan_c2r, (hipfftComplex*)d_cx, (hipfftReal*)d_cx);
      }

#endif
    // Wait for execution to finish
    hip_error = hipDeviceSynchronize();

    // testPlan(plan_r2c, Nbytes, (hipfftReal*)d_cx, (hipfftReal*)cy);

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
    result = hipfftDestroy(plan_r2c);
#if 1
    result = hipfftDestroy(plan_c2r);
#endif

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
