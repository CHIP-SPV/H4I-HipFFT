# Copyright 2021-2023 UT-Battelle
# See LICENSE.txt in the root of the source distribution for license info.
cmake_minimum_required(VERSION 3.23)
include (${CMAKE_SOURCE_DIR}/CMake/HipFFTVersion.cmake)
project(HipFFT
    VERSION ${HipFFT_VERSION}
    LANGUAGES CXX)


# We will use HIP in some fashion, no matter which platform
# we're targeting or what parts of the software we're building.
find_package(HIP REQUIRED)
# When using CHIP-SPV, CMake's method for finding a HIP compiler
# seems to want to find ROCm.  Since ROCm isn't in play, CMake
# fails to enable support for HIP as a first class language.
#enable_language(HIP)

# Ensure our users and installed tests will be able to find our
# dependency libraries.
include(GNUInstallDirs)
file(RELATIVE_PATH relRPath
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relRPath})

# Define a target capturing common configuration settings.
# Although we use 'add_library' for this, it is not a library - 
# just a CMake target with a collection of properties set the
# way we want.
# Unfortunately, it doesn't seem to be possible to set all
# of the properties we want on this target and have them
# be inherited by targets that "link" it.
# In particular, we can't set a few C++ standards properties
# and so either have to set them globally or on every target.
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(HipFFTCommonConfig INTERFACE)
target_compile_features(HipFFTCommonConfig
    INTERFACE
        cxx_std_17
    )

add_subdirectory(src)

# Build tests if desired.
#include(CTest)
#if(BUILD_TESTING)
#    add_subdirectory(tests)
#endif()

