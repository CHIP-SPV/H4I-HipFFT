# Add the test executable
set(CMAKE_CXX_COMPILER clang++)
add_executable(hipfft_complex_1d hipfft_complex_1d.cpp)

target_include_directories(hipfft_complex_1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(hipfft_complex_1d PRIVATE -fsycl)
target_link_options(hipfft_complex_1d PRIVATE -fsycl)
# Link libraries
target_link_libraries(hipfft_complex_1d PRIVATE
    hipfft
    hip::device
)

# Add the test to CTest
add_test(NAME hipfft_complex_1d COMMAND hipfft_complex_1d)
