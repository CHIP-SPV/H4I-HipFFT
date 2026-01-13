# =============================================================================
# CMake build system files
#
# Copyright (c) 2021-24 chipStar developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# =============================================================================

cmake_minimum_required(VERSION 3.20)

include(GNUInstallDirs)
# Check if INTEL_COMPILER_PATH was passed to CMake invocation, if not, look for icpx in PATH
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    find_program(INTEL_COMPILER_PATH icpx)
    if(INTEL_COMPILER_PATH)
        set(CMAKE_CXX_COMPILER ${INTEL_COMPILER_PATH})
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
        message(STATUS "Found Intel C++ Compiler: ${INTEL_COMPILER_PATH}")
    else()
        message(WARNING "Intel C++ Compiler (icpx) not found. Using default compiler.")
    endif()
endif()

# Must be after setting CMAKE_CXX_COMPILER to intel otherwise it will go into an infinite loop
project(hipFFT
    VERSION 1.0.0
    LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
    
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




# Select SPIR-V platform
add_compile_definitions(__HIP_PLATFORM_SPIRV__)

# Find HIP
if(STANDALONE_BUILD)
  # Find externally installed H4I-MKLShim
  find_package(MKLShim CONFIG REQUIRED)
  include_directories(${H4I_MKLShim_INCLUDE_DIR})
  
  find_package(hip CONFIG REQUIRED)
  include_directories($ENV{HIP_PATH}/include)
endif()


# Create the hipFFT library
add_library(hipfft SHARED
    src/hipfft.cpp
)

# Link against MKL and other required libraries
target_link_libraries(hipfft
    PUBLIC
        hip::host
        H4I::MKLShim
    PRIVATE
        MKL::MKL
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
target_include_directories(hipfft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

include(CTest)
add_subdirectory(tests)