include(GoogleTest)

add_executable(broadcast_shape_test broadcast_shape_test.cpp)
target_link_libraries(broadcast_shape_test PRIVATE helix GTest::gtest_main)
add_test(NAME BroadcastShapeTest COMMAND broadcast_shape_test)

add_executable(broadcast_stride_test broadcast_stride_test.cpp)
target_link_libraries(broadcast_stride_test PRIVATE helix GTest::gtest_main)
add_test(NAME BroadcastStrideTest COMMAND broadcast_stride_test)

add_executable(ops_test ops_test.cpp)
target_link_libraries(ops_test PRIVATE helix GTest::gtest_main)
add_test(NAME OpsTest COMMAND ops_test)

add_executable(matmul_test matmul_test.cpp)
target_link_libraries(matmul_test PRIVATE helix GTest::gtest_main)
add_test(NAME MatmulTest COMMAND matmul_test)

add_executable(manipulation_test manipulation_test.cpp)
target_link_libraries(manipulation_test PRIVATE helix GTest::gtest_main)
add_test(NAME ManipulationTest COMMAND manipulation_test)

add_executable(reduce_test reduce_test.cpp)
target_link_libraries(reduce_test PRIVATE helix GTest::gtest_main)
add_test(NAME ReduceTest COMMAND reduce_test)
add_executable(core_test core_test.cpp)
target_link_libraries(core_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(core_test)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(tensor_test)

add_executable(autograd_test autograd_test.cpp)
target_link_libraries(autograd_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(autograd_test)

add_executable(autograd_ops_test autograd_ops_test.cpp grad_check.cpp)
target_link_libraries(autograd_ops_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(autograd_ops_test)

# Neural Network layer tests
add_executable(module_test module_test.cpp)
target_link_libraries(module_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(module_test)

add_executable(linear_test linear_test.cpp grad_check.cpp)
target_link_libraries(linear_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(linear_test)

add_executable(relu_test relu_test.cpp grad_check.cpp)
target_link_libraries(relu_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(relu_test)

add_executable(loss_test loss_test.cpp grad_check.cpp)
target_link_libraries(loss_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(loss_test)

add_executable(sequential_test sequential_test.cpp)
target_link_libraries(sequential_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(sequential_test)

add_executable(nn_integration_test nn_integration_test.cpp grad_check.cpp)
target_link_libraries(nn_integration_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(nn_integration_test)
