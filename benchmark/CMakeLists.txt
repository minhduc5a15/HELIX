add_executable(matmul_benchmark matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE helix)

add_executable(tensor_ops_benchmark tensor_ops_benchmark.cpp)
target_link_libraries(tensor_ops_benchmark PRIVATE helix)

add_executable(nn_benchmark nn_benchmark.cpp)
target_link_libraries(nn_benchmark PRIVATE helix)

add_executable(training_benchmark training_benchmark.cpp)
target_link_libraries(training_benchmark PRIVATE helix)

add_executable(bmm_benchmark bmm_benchmark.cpp)
target_link_libraries(bmm_benchmark PRIVATE helix)
target_include_directories(bmm_benchmark PRIVATE ${PROJECT_SOURCE_DIR}/src)
