add_executable(hipfft_missing_symbols_test hipfft_missing_symbols.cpp)

target_include_directories(hipfft_missing_symbols_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hipfft_missing_symbols_test PRIVATE hipfft)

add_test(NAME hipfft_missing_symbols COMMAND hipfft_missing_symbols_test)
set_tests_properties(hipfft_missing_symbols PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED")
