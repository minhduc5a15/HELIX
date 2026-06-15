include(GoogleTest)

add_executable(value_test value_test.cpp)
target_link_libraries(value_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(value_test)

add_executable(gradient_check_test gradient_check_test.cpp)
target_link_libraries(gradient_check_test PRIVATE helix GTest::gtest_main)
gtest_discover_tests(gradient_check_test)
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
