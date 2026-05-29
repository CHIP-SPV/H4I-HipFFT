# Standalone consumer project that exercises find_package(hipfft CONFIG).
# This is intentionally NOT added via add_subdirectory from the top-level
# build; CI configures it separately against an installed hipfft so that
# the install-time CMake package config is what's actually tested.
cmake_minimum_required(VERSION 3.20)

project(hipfft_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(__HIP_PLATFORM_SPIRV__)

find_package(hip CONFIG REQUIRED)
find_package(hipfft CONFIG REQUIRED)

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE hip::hipfft)
