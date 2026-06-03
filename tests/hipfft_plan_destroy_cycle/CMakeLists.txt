add_executable(hipfft_plan_destroy_cycle_test hipfft_plan_destroy_cycle.cpp)

target_include_directories(hipfft_plan_destroy_cycle_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(hipfft_plan_destroy_cycle_test PRIVATE hipfft hip::host)

add_test(NAME hipfft_plan_destroy_cycle COMMAND hipfft_plan_destroy_cycle_test)
set_tests_properties(hipfft_plan_destroy_cycle PROPERTIES
    PASS_REGULAR_EXPRESSION "PASSED"
    TIMEOUT 60)
