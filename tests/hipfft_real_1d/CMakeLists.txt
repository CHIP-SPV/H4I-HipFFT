# Add the test executable
add_executable(hipfft_real_1d_test hipfft_real_1d.cpp)

target_include_directories(hipfft_real_1d_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
# Link libraries
target_link_libraries(hipfft_real_1d_test PRIVATE hipfft)

# Add the test to CTest
add_test(NAME hipfft_real_1d COMMAND hipfft_real_1d_test)
