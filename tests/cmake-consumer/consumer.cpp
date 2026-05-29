// Minimal consumer that links against an installed hipfft via
// find_package(hipfft CONFIG). The purpose is to verify the install-time
// CMake package config exposes a usable hip::hipfft target with correct
// include directories and link dependencies. No real FFT work is done.
#include <cstdlib>
#include <iostream>

#include <hipfft.h>

int main()
{
    hipfftHandle plan = nullptr;
    hipfftResult result = hipfftCreate(&plan);
    if (result != HIPFFT_SUCCESS) {
        std::cerr << "FAILED: hipfftCreate returned " << result << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "PASSED: find_package(hipfft CONFIG) consumer linked and ran"
              << std::endl;
    return EXIT_SUCCESS;
}
