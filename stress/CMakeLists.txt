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

add_executable(test_autograd_inplace test_autograd_inplace.cpp)
target_link_libraries(test_autograd_inplace PRIVATE helix GTest::gtest_main)
gtest_discover_tests(test_autograd_inplace)


add_executable(test_openmp_race test_openmp_race.cpp)
target_link_libraries(test_openmp_race PRIVATE helix GTest::gtest_main)
gtest_discover_tests(test_openmp_race)

add_executable(test_allocator_leak test_allocator_leak.cpp)
target_link_libraries(test_allocator_leak PRIVATE helix GTest::gtest_main)
gtest_discover_tests(test_allocator_leak)

add_executable(test_broadcast_negative_strides test_broadcast_negative_strides.cpp)
target_link_libraries(test_broadcast_negative_strides PRIVATE helix GTest::gtest_main)
gtest_discover_tests(test_broadcast_negative_strides)

add_executable(test_graph_leak test_graph_leak.cpp)
target_link_libraries(test_graph_leak PRIVATE helix GTest::gtest_main)
gtest_discover_tests(test_graph_leak)
