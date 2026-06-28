add_executable(xor_mlp xor_mlp.cpp)
target_link_libraries(xor_mlp PRIVATE helix)

add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE helix)
