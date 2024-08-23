# Add the test executable
add_executable(hipfft_complex_1d hipfft_complex_1d.cpp)

target_include_directories(hipfft_complex_1d PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
# Link libraries
target_link_libraries(hipfft_complex_1d PRIVATE hipfft)

# Add the test to CTest
add_test(NAME hipfft_complex_1d COMMAND hipfft_complex_1d)
