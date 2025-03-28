#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "hip/hip_runtime.h"
#include "hip/hip_interop.h"
#include "hipfft/hipfft.h"

#define HIP_CHECK(stat)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = stat;                                          \
        if(err != hipSuccess)                                           \
        {                                                               \
            std::cerr << "HIP error: " << hipGetErrorString(err)        \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

#define HIPFFT_CHECK(stat)                                              \
    do                                                                  \
    {                                                                   \
        hipfftResult err = stat;                                        \
        if(err != HIPFFT_SUCCESS)                                       \
        {                                                               \
            std::cerr << "hipFFT error: " << err                        \
                      << " at line " << __LINE__                        \
                      << std::endl;                                     \
            exit(err);                                                  \
        }                                                               \
    } while(0)

int main() {
    std::cout << "======== H4I-hipfft Basic Test ========" << std::endl;
    
    // Test plan creation
    std::cout << "Testing hipFFT plan creation..." << std::endl;
    hipfftHandle plan = nullptr;
    HIPFFT_CHECK(hipfftCreate(&plan));
    
    if (plan) {
        std::cout << "Plan creation successful!" << std::endl;
    } else {
        std::cerr << "Failed to create plan!" << std::endl;
        return -1;
    }

    // Test stream creation and setting
    std::cout << "Testing hipFFT stream handling..." << std::endl;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIPFFT_CHECK(hipfftSetStream(plan, stream));
    std::cout << "Stream handling successful!" << std::endl;

    // Test a simple 1D FFT transformation
    std::cout << "Testing 1D FFT R2C and C2R transformations..." << std::endl;
    const int nx = 8;
    size_t workSize = 0;
    
    // Create 1D FFT plan
    HIPFFT_CHECK(hipfftMakePlan1d(plan, nx, HIPFFT_R2C, 1, &workSize));
    std::cout << "1D FFT plan created, work size: " << workSize << " bytes" << std::endl;
    
    // Prepare input/output data
    std::vector<float> h_signal(nx);
    // Initialize with a simple signal: a single sine period
    for (int i = 0; i < nx; i++) {
        h_signal[i] = std::sin(2.0f * M_PI * i / nx);
    }
    
    std::cout << "Input signal: ";
    for (int i = 0; i < nx; i++) {
        std::cout << h_signal[i] << " ";
    }
    std::cout << std::endl;
    
    // Allocate GPU memory
    float* d_signal;
    hipfftComplex* d_freq;
    float* d_recovered;
    
    HIP_CHECK(hipMalloc(&d_signal, nx * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_freq, (nx/2 + 1) * sizeof(hipfftComplex)));
    HIP_CHECK(hipMalloc(&d_recovered, nx * sizeof(float)));
    
    // Copy input data to device
    HIP_CHECK(hipMemcpy(d_signal, h_signal.data(), nx * sizeof(float), hipMemcpyHostToDevice));
    
    // Execute forward FFT (R2C)
    HIPFFT_CHECK(hipfftExecR2C(plan, d_signal, d_freq));
    
    // Get frequency domain data
    std::vector<hipfftComplex> h_freq((nx/2) + 1);
    HIP_CHECK(hipMemcpy(h_freq.data(), d_freq, (nx/2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost));
    
    std::cout << "Frequency domain data: ";
    for (int i = 0; i < nx/2 + 1; i++) {
        std::cout << "(" << h_freq[i].x << "," << h_freq[i].y << ") ";
    }
    std::cout << std::endl;
    
    // Execute inverse FFT (C2R)
    HIPFFT_CHECK(hipfftExecC2R(plan, d_freq, d_recovered));
    
    // Get recovered signal
    std::vector<float> h_recovered(nx);
    HIP_CHECK(hipMemcpy(h_recovered.data(), d_recovered, nx * sizeof(float), hipMemcpyDeviceToHost));
    
    // Scale the output (C2R doesn't normalize)
    for (int i = 0; i < nx; i++) {
        h_recovered[i] /= nx;
    }
    
    std::cout << "Recovered signal (after scaling): ";
    for (int i = 0; i < nx; i++) {
        std::cout << h_recovered[i] << " ";
    }
    std::cout << std::endl;
    
    // Validate results
    bool success = true;
    float max_error = 0.0f;
    for (int i = 0; i < nx; i++) {
        float error = std::abs(h_recovered[i] - h_signal[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-5) {
            success = false;
        }
    }
    
    std::cout << "Maximum error: " << max_error << std::endl;
    if (success) {
        std::cout << "FFT operations successful!" << std::endl;
    } else {
        std::cerr << "FFT operations error: signal recovery mismatch!" << std::endl;
    }
    
    // Clean up
    HIP_CHECK(hipFree(d_signal));
    HIP_CHECK(hipFree(d_freq));
    HIP_CHECK(hipFree(d_recovered));
    HIP_CHECK(hipStreamDestroy(stream));
    HIPFFT_CHECK(hipfftDestroy(plan));
    
    std::cout << "======== Test Complete ========" << std::endl;
    return 0;
} 