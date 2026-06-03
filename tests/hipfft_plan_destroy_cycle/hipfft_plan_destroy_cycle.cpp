// Regression test: repeated hipfftPlan3d / exec / hipfftDestroy in a
// single process must not crash.
//
// Reproduces a SIGSEGV observed when OpenMM's benchmark.py ran two
// PME-using tests back-to-back (rf,pme,apoa1pme,...) on Aurora PVC:
// the second hipfftPlan3d would segfault inside oneMKL's gpu_commit
// -> sycl::handler::finalize, called from
// H4I::MKLShim::createFFTDescriptorSR.
//
// Root cause: hipfftDestroy released the FFT descriptors but never
// called H4I::MKLShim::Destroy(plan->ctxt), so the per-queue Context
// stayed in MKLShim::context_tbl (keyed by the underlying L0 queue
// handle pointer). When the chipStar runtime later recycled that L0
// queue handle for a freshly created hipStream_t, the second
// hipfftCreate found the stale Context and returned it; oneMKL then
// committed against a SYCL queue whose backing L0 queue had been
// destroyed.
//
// Before the fix: this test segfaults on iteration i == 1.
// After the fix: prints "PASSED".

#include <cstdlib>
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include "hipfftHandle.h"
#include "hipfft.h"

int main() {
    const int N = 64;
    const size_t in_count = static_cast<size_t>(N) * N * N;
    const size_t out_count = static_cast<size_t>(N) * N * (N / 2 + 1);
    std::vector<float> host_in(in_count, 1.0f);

    for (int i = 0; i < 3; i++) {
        std::cout << "iter " << i << ": create stream + plan" << std::endl;

        hipStream_t S = nullptr;
        if (hipStreamCreate(&S) != hipSuccess) {
            std::cerr << "FAILED iter " << i << ": hipStreamCreate" << std::endl;
            return EXIT_FAILURE;
        }

        hipfftHandle plan = nullptr;
        if (hipfftPlan3d(&plan, N, N, N, HIPFFT_R2C) != HIPFFT_SUCCESS ||
            plan == nullptr) {
            std::cerr << "FAILED iter " << i << ": hipfftPlan3d" << std::endl;
            return EXIT_FAILURE;
        }
        if (hipfftSetStream(plan, S) != HIPFFT_SUCCESS) {
            std::cerr << "FAILED iter " << i << ": hipfftSetStream" << std::endl;
            return EXIT_FAILURE;
        }

        float *d_in = nullptr;
        hipfftComplex *d_out = nullptr;
        hipMalloc(&d_in, in_count * sizeof(float));
        hipMalloc(&d_out, out_count * sizeof(hipfftComplex));
        hipMemcpyAsync(d_in, host_in.data(), in_count * sizeof(float),
                       hipMemcpyHostToDevice, S);
        if (hipfftExecR2C(plan, d_in, d_out) != HIPFFT_SUCCESS) {
            std::cerr << "FAILED iter " << i << ": hipfftExecR2C" << std::endl;
            return EXIT_FAILURE;
        }
        hipStreamSynchronize(S);

        hipfftDestroy(plan);
        hipFree(d_in);
        hipFree(d_out);
        hipStreamDestroy(S);
    }

    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
