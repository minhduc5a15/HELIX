add_executable(matmul_benchmark matmul_benchmark.cpp)
target_link_libraries(matmul_benchmark PRIVATE helix benchmark::benchmark benchmark::benchmark_main)
