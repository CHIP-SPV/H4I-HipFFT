cmake_minimum_required(VERSION 3.15)

# Add basic test
add_executable(fft_basic_test fft_basic_test.cpp)
target_link_libraries(fft_basic_test PRIVATE hip::device hipfft)

# Install the sample in bin directory
install(TARGETS fft_basic_test DESTINATION bin)

# Add the test to CTest
add_test(NAME hipfft_basic_context_test COMMAND fft_basic_test) 