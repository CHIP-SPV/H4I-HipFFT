cmake_minimum_required(VERSION 3.0.0)
project(hipfft VERSION 0.1.0)

set (CMAKE_GENERATOR "Unix Makefiles" CACHE INTERNAL "" FORCE)


include(CTest)
enable_testing()
add_compile_definitions(__HIP_PLATFORM_SPIRV__)
include_directories("include"   
"/gpfs/jlse-fs0/users/pvelesko/install/HIP/clang15/chip-spv-latest/include"
"/soft/compilers/oneapi-2023.0.0/mkl/2023.0.0/include"
"/soft/compilers/oneapi-2023.0.0/compiler/2023.0.0/linux/include"
"/soft/compilers/oneapi-2023.0.0/compiler/2023.0.0/linux/include/sycl"
)
add_library(hipfft hipfft.cpp)
#add_compile_options("-x hip")
set_property(TARGET hipfft PROPERTY CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
