// Regression test for hipfftCreate() returning a handle with a null context.
//
// hipfftCreate() calls hipGetBackendNativeHandles() which returns nHandles=5,
// where handles[0] is the backend name and handles[1..4] are native
// platform/device/context/queue handles. A previous bug shifted the array
// and decremented nHandles to 4 before calling MKLShim::Create(), which
// requires numOfHandles >= 5 and returned nullptr on this condition,
// leaving plan->ctxt == nullptr and causing a segfault on first use.
#include <cstdlib>
#include <iostream>
#include "hipfftHandle.h"
#include "hipfft.h"

int main()
{
    hipfftHandle plan = nullptr;
    hipfftResult result = hipfftCreate(&plan);

    if (result != HIPFFT_SUCCESS) {
        std::cerr << "FAILED: hipfftCreate returned " << result << std::endl;
        return EXIT_FAILURE;
    }
    if (plan == nullptr) {
        std::cerr << "FAILED: hipfftCreate returned null handle" << std::endl;
        return EXIT_FAILURE;
    }
    if (plan->ctxt == nullptr) {
        std::cerr << "FAILED: hipfftCreate handle has null MKLShim context" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
