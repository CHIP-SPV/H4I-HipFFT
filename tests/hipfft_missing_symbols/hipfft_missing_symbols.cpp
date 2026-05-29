// Reproduces issue #8: missing hipfftPlan3d, hipfftMakePlan3d, hipfftDestroy,
// hipfftSetStream, hipfftGetVersion, and the double-precision Exec variants
// (Z2Z, D2Z, Z2D).  Today this fails to link because libhipfft.so doesn't
// export these symbols.  After the fix it should run end-to-end.
#include <cstdlib>
#include <iostream>
#include <vector>
#include "hip/hip_runtime.h"
#include "hipfftHandle.h"
#include "hipfft.h"

#define CHECK(call) do { \
    hipfftResult r = (call); \
    if (r != HIPFFT_SUCCESS) { \
        std::cerr << "FAILED: " #call " returned " << r << std::endl; \
        return EXIT_FAILURE; \
    } \
} while (0)

int main() {
    int version = 0;
    CHECK(hipfftGetVersion(&version));
    if (version <= 0) {
        std::cerr << "FAILED: hipfftGetVersion returned non-positive " << version << std::endl;
        return EXIT_FAILURE;
    }

    const int nx = 8, ny = 8, nz = 8;

    // hipfftPlan3d: single-precision C2C
    hipfftHandle plan3d_c2c = nullptr;
    CHECK(hipfftPlan3d(&plan3d_c2c, nx, ny, nz, HIPFFT_C2C));

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::cerr << "FAILED: hipStreamCreate" << std::endl;
        return EXIT_FAILURE;
    }
    CHECK(hipfftSetStream(plan3d_c2c, stream));

    // 3D double-complex round trip via Z2Z
    hipfftHandle plan3d_z2z = nullptr;
    CHECK(hipfftPlan3d(&plan3d_z2z, nx, ny, nz, HIPFFT_Z2Z));

    const size_t nelem = (size_t)nx * ny * nz;
    std::vector<hipfftDoubleComplex> host_in(nelem), host_out(nelem);
    for (size_t i = 0; i < nelem; i++) {
        host_in[i].x = (double)(i % 7) - 3.0;
        host_in[i].y = (double)(i % 5) - 2.0;
    }
    hipfftDoubleComplex *d_a = nullptr, *d_b = nullptr;
    if (hipMalloc(&d_a, nelem * sizeof(hipfftDoubleComplex)) != hipSuccess ||
        hipMalloc(&d_b, nelem * sizeof(hipfftDoubleComplex)) != hipSuccess) {
        std::cerr << "FAILED: hipMalloc" << std::endl;
        return EXIT_FAILURE;
    }
    hipMemcpy(d_a, host_in.data(), nelem * sizeof(hipfftDoubleComplex), hipMemcpyHostToDevice);
    CHECK(hipfftExecZ2Z(plan3d_z2z, d_a, d_b, HIPFFT_FORWARD));
    CHECK(hipfftExecZ2Z(plan3d_z2z, d_b, d_a, HIPFFT_BACKWARD));
    hipMemcpy(host_out.data(), d_a, nelem * sizeof(hipfftDoubleComplex), hipMemcpyDeviceToHost);

    // Round-trip with backend normalization: input ≈ output / N
    double tol = 1e-6 * (double)nelem;
    for (size_t i = 0; i < nelem; i++) {
        double rx = host_out[i].x / (double)nelem;
        double ry = host_out[i].y / (double)nelem;
        if (std::abs(rx - host_in[i].x) > tol || std::abs(ry - host_in[i].y) > tol) {
            std::cerr << "FAILED: Z2Z round-trip at i=" << i
                      << " got (" << rx << "," << ry
                      << ") expected (" << host_in[i].x << "," << host_in[i].y << ")" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // 1D double real round trip via D2Z + Z2D.  Use 1D where the
    // conjugate-even packed layout is unambiguous; the 3D layout
    // (cuFFT full-complex vs oneMKL COMPLEX_REAL) is a separate
    // storage-config concern out of scope for this PR.
    const int n1d = 16;
    hipfftHandle plan1d_d2z = nullptr, plan1d_z2d = nullptr;
    CHECK(hipfftPlan1d(&plan1d_d2z, n1d, HIPFFT_D2Z, 1));
    CHECK(hipfftPlan1d(&plan1d_z2d, n1d, HIPFFT_Z2D, 1));

    double *d_r = nullptr;
    hipfftDoubleComplex *d_c = nullptr;
    hipMalloc(&d_r, (n1d + 2) * sizeof(double));
    hipMalloc(&d_c, (n1d / 2 + 1) * sizeof(hipfftDoubleComplex));
    std::vector<double> host_r_in(n1d), host_r_out(n1d);
    for (int i = 0; i < n1d; i++) host_r_in[i] = (double)(i % 11) - 5.0;
    hipMemcpy(d_r, host_r_in.data(), n1d * sizeof(double), hipMemcpyHostToDevice);
    CHECK(hipfftExecD2Z(plan1d_d2z, d_r, d_c));
    CHECK(hipfftExecZ2D(plan1d_z2d, d_c, d_r));
    hipMemcpy(host_r_out.data(), d_r, n1d * sizeof(double), hipMemcpyDeviceToHost);
    double tol1d = 1e-6 * (double)n1d;
    for (int i = 0; i < n1d; i++) {
        double r = host_r_out[i] / (double)n1d;
        if (std::abs(r - host_r_in[i]) > tol1d) {
            std::cerr << "FAILED: 1D D2Z/Z2D round-trip at i=" << i
                      << " got " << r << " expected " << host_r_in[i] << std::endl;
            return EXIT_FAILURE;
        }
    }

    // hipfftDestroy should clean up all four plans without crashing
    CHECK(hipfftDestroy(plan3d_c2c));
    CHECK(hipfftDestroy(plan3d_z2z));
    CHECK(hipfftDestroy(plan1d_d2z));
    CHECK(hipfftDestroy(plan1d_z2d));

    hipFree(d_a); hipFree(d_b); hipFree(d_r); hipFree(d_c);
    hipStreamDestroy(stream);

    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
