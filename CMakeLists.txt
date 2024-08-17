# Copyright [YEAR] [YOUR ORGANIZATION]
# See LICENSE.txt in the root of the source distribution for license info.

cmake_minimum_required(VERSION 3.20)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(STANDALONE_BUILD ON)
else()
    set(STANDALONE_BUILD OFF)
endif()

# set the default CMAKE_INSTALL_PREFIX to the current directory/install
if(STANDALONE_BUILD AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE PATH "..." FORCE)
endif()

# print where the library will be installed
message(STATUS "hipFFT will be installed to: ${CMAKE_INSTALL_PREFIX}")

project(hipFFT
    VERSION 1.0.0
    LANGUAGES CXX)

# Check if INTEL_COMPILER_PATH was passed to CMake invocation, if not, look for icpx in PATH
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    find_program(INTEL_COMPILER_PATH icpx)
    if(INTEL_COMPILER_PATH)
        set(CMAKE_CXX_COMPILER ${INTEL_COMPILER_PATH} CACHE STRING "C++ compiler" FORCE)
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM" CACHE STRING "C++ compiler ID" FORCE)
        message(STATUS "Found Intel C++ Compiler: ${INTEL_COMPILER_PATH}")
    else()
        message(WARNING "Intel C++ Compiler (icpx) not found. Using default compiler.")
    endif()
endif()

# Define __HIP_PLATFORM_SPIRV__
add_compile_definitions(__HIP_PLATFORM_SPIRV__)


# Find HIP
if(STANDALONE_BUILD)
  # Build H4I-MKLShim from submodule first
  add_subdirectory(H4I-MKLShim)
  
  # Find hip package, but exclude its H4I-MKLShim
  set(CMAKE_IGNORE_PATH ${CMAKE_IGNORE_PATH} ${hip_DIR}/../H4I-MKLShim)
  find_package(hip CONFIG REQUIRED)
  
  # Explicitly set the path to our built H4I-MKLShim
  set(H4I_MKLShim_DIR ${CMAKE_CURRENT_BINARY_DIR}/H4I-MKLShim)
endif()

# Ensure our users and installed tests will be able to find our
# dependency libraries.
include(GNUInstallDirs)
file(RELATIVE_PATH relRPath
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relRPath})

# Define a target capturing common configuration settings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_library(hipFFTCommonConfig INTERFACE)
target_compile_features(hipFFTCommonConfig
    INTERFACE
        cxx_std_17
    )

# Add include directories
target_include_directories(hipFFTCommonConfig
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        /home/ac.nichols/H4I/INSTALLATION/MKLSHIM_DEVELOP/include
)

# Add compile definitions
target_compile_definitions(hipFFTCommonConfig
    INTERFACE
        __HIP_PLATFORM_SPIRV__
)

# Create the hipFFT library
add_library(hipfft SHARED
    src/hipfft.cpp
)

# Link against MKL and other required libraries
target_link_libraries(hipfft
    PRIVATE
        MKL::MKL
        ${CMAKE_DL_LIBS}
        CHIP
        MKLShim
)

# Set properties for the hipfft target
set_target_properties(hipfft PROPERTIES
    OUTPUT_NAME "hipfft"
    POSITION_INDEPENDENT_CODE ON
)

# Modify the hipfft target to use Intel's SYCL flag
target_compile_options(hipfft PRIVATE -fsycl)
target_link_options(hipfft PRIVATE -fsycl)

# Add the directory containing hipfftHandle.h to the include directories
target_include_directories(hipfft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Installation rules
install(TARGETS hipfft
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

# Build tests if desired
include(CTest)
if(BUILD_TESTING)
    # Add test subdirectory if you have tests
    # add_subdirectory(test)
endif()