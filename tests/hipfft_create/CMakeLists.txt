add_executable(hipfft_create_test hipfft_create.cpp)

target_include_directories(hipfft_create_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hipfft_create_test PRIVATE hipfft)

add_test(NAME hipfft_create COMMAND hipfft_create_test)
set_tests_properties(hipfft_create PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED")
