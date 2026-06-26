include(GoogleTest)

include_directories(../tests)

add_executable(module_stress module_stress.cpp)
target_link_libraries(module_stress PRIVATE helix GTest::gtest_main)
gtest_discover_tests(module_stress)

add_executable(linear_stress linear_stress.cpp ../tests/grad_check.cpp)
target_link_libraries(linear_stress PRIVATE helix GTest::gtest_main)
gtest_discover_tests(linear_stress)

add_executable(relu_stress relu_stress.cpp ../tests/grad_check.cpp)
target_link_libraries(relu_stress PRIVATE helix GTest::gtest_main)
gtest_discover_tests(relu_stress)

add_executable(sequential_stress sequential_stress.cpp)
target_link_libraries(sequential_stress PRIVATE helix GTest::gtest_main)
gtest_discover_tests(sequential_stress)

add_executable(nn_integration_stress nn_integration_stress.cpp ../tests/grad_check.cpp)
target_link_libraries(nn_integration_stress PRIVATE helix GTest::gtest_main)
gtest_discover_tests(nn_integration_stress)
