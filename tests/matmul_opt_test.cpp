#include <gtest/gtest.h>

#include <vector>

#include "helix.hpp"

using namespace helix;

// Helper function to create a matrix with sequential values
Tensor create_sequential_matrix(size_t rows, size_t cols) {
    std::vector<float> data(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    return Tensor(data, {rows, cols});
}

// Identity Matrix Test
TEST(MatMulOptTest, IdentityMatrix) {
    size_t size = 127;
    auto A = create_sequential_matrix(size, size);

    std::vector<float> id_data(size * size, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        id_data[i * size + i] = 1.0f;
    }
    auto I = Tensor(id_data, {size, size});

    auto C = A.matmul(I);

    const float* a_ptr = A.data_ptr();
    const float* c_ptr = C.data_ptr();

    for (size_t i = 0; i < size * size; ++i) {
        EXPECT_NEAR(a_ptr[i], c_ptr[i], 1e-4);
    }
}

// Zero Matrix Test
TEST(MatMulOptTest, ZeroMatrix) {
    size_t m = 135;
    size_t k = 222;
    size_t n = 79;

    auto A = create_sequential_matrix(m, k);
    std::vector<float> zeros(k * n, 0.0f);
    auto Z = Tensor(zeros, {k, n});

    auto C = A.matmul(Z);

    const float* c_ptr = C.data_ptr();

    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_EQ(c_ptr[i], 0.0f);
    }
}

// Odd Shapes Test
TEST(MatMulOptTest, OddShapes) {
    size_t m = 13;
    size_t k = 17;
    size_t n = 19;

    auto A = create_sequential_matrix(m, k);
    auto B = create_sequential_matrix(k, n);

    auto C = A.matmul(B);

    const float* a_ptr = A.data_ptr();
    const float* b_ptr = B.data_ptr();
    const float* c_ptr = C.data_ptr();

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a_ptr[i * k + l] * b_ptr[l * n + j];
            }
            EXPECT_NEAR(c_ptr[i * n + j], sum, 1e-4);
        }
    }
}

// Large Matrices Test (Ensures OpenMP/AVX paths work correctly on larger than threshold)
TEST(MatMulOptTest, LargeMatrices) {
    size_t m = 257;
    size_t k = 133;
    size_t n = 311;

    std::vector<float> a_data(m * k, 1.0f);
    std::vector<float> b_data(k * n, 2.0f);

    auto A = Tensor(a_data, {m, k});
    auto B = Tensor(b_data, {k, n});

    auto C = A.matmul(B);

    const float* c_ptr = C.data_ptr();

    float expected = 1.0f * 2.0f * static_cast<float>(k);
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_NEAR(c_ptr[i], expected, 1e-4);
    }
}
