# Add the test executable
set(CMAKE_CXX_COMPILER clang++)
add_executable(hipfft_real_1d_test hipfft_real_1d.cpp)

target_include_directories(hipfft_real_1d_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(hipfft_real_1d_test PRIVATE -fsycl)
target_link_options(hipfft_real_1d_test PRIVATE -fsycl)
# Link libraries
target_link_libraries(hipfft_real_1d_test PRIVATE
    hipfft
    hip::device
)

# Add the test to CTest
add_test(NAME hipfft_real_1d COMMAND hipfft_real_1d_test)
