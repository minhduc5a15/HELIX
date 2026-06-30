#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "backend/cpu_backend.hpp"
#include "helix.hpp"

using namespace helix;

// ---------------------------------------------------------
// Helpers
// ---------------------------------------------------------

Tensor create_sequential_matrix(size_t rows, size_t cols) {
    std::vector<float> data(rows * cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    return Tensor(data, {rows, cols});
}

Tensor create_identity_matrix(size_t size) {
    std::vector<float> data(size * size, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        data[i * size + i] = 1.0f;
    }
    return Tensor(data, {size, size});
}

// ---------------------------------------------------------
// 1. Happy Path & 2. Edge Cases (Parameterized Sizes)
// ---------------------------------------------------------

class MatMulOptTest : public ::testing::TestWithParam<size_t> {};

// Provide the specific sizes requested for unit tests: {63, 127, 255, 1023}
INSTANTIATE_TEST_SUITE_P(Sizes, MatMulOptTest, ::testing::Values(1, 7, 8, 9, 15, 16, 17, 63, 127, 255, 1023));

TEST_P(MatMulOptTest, DispatcherHappyPath) {
    size_t size = GetParam();
    auto A = create_sequential_matrix(size, size);
    auto I = create_identity_matrix(size);

    auto C = A.matmul(I);

    const float* a_ptr = A.data_ptr();
    const float* c_ptr = C.data_ptr();
    for (size_t i = 0; i < size * size; ++i) {
        EXPECT_NEAR(a_ptr[i], c_ptr[i], 1e-3);
    }
}

TEST_P(MatMulOptTest, DirectKernelEdgeCases) {
    size_t size = GetParam();

    std::vector<float> A_data(size * size, 1.0f);
    std::vector<float> B_T_data(size * size, 2.0f);
    std::vector<float> C_data(size * size, 0.0f);

    // Naive Kernel
    std::fill(C_data.begin(), C_data.end(), 0.0f);
    CPUBackend::matmul(A_data.data(), B_T_data.data(), C_data.data(), size, size, size, MatMulStrategy::Naive);
    EXPECT_FLOAT_EQ(C_data[0], 2.0f * size);

    // Blocked Kernel
    std::fill(C_data.begin(), C_data.end(), 0.0f);
    CPUBackend::matmul(A_data.data(), B_T_data.data(), C_data.data(), size, size, size, MatMulStrategy::Blocked);
    EXPECT_FLOAT_EQ(C_data[0], 2.0f * size);

    // AVX2 Kernel
    std::fill(C_data.begin(), C_data.end(), 0.0f);
    CPUBackend::matmul(A_data.data(), B_T_data.data(), C_data.data(), size, size, size, MatMulStrategy::AVX2);
    EXPECT_FLOAT_EQ(C_data[0], 2.0f * size);

    // OpenMP Kernel
    std::fill(C_data.begin(), C_data.end(), 0.0f);
    CPUBackend::matmul(A_data.data(), B_T_data.data(), C_data.data(), size, size, size, MatMulStrategy::OpenMP);
    EXPECT_FLOAT_EQ(C_data[0], 2.0f * size);
}

// ---------------------------------------------------------
// 3. Error Handling
// ---------------------------------------------------------

TEST(MatMulErrorTest, IncompatibleShapes) {
    auto A = create_sequential_matrix(10, 20);
    auto B = create_sequential_matrix(30, 40);  // Inner dims 20 and 30 do not match

    EXPECT_THROW(A.matmul(B), std::invalid_argument);
}

TEST(MatMulErrorTest, Non2DShapes) {
    Tensor A(std::vector<float>(10), {10});       // 1D
    Tensor B = create_sequential_matrix(10, 20);  // 2D

    // matmul strictly expects 2D tensors in current HELIX implementation
    EXPECT_THROW(A.matmul(B), std::invalid_argument);
}

// ---------------------------------------------------------
// 4. Stress & Scale
// ---------------------------------------------------------

TEST(MatMulStressTest, RectangularOddShapes_ZeroMatrix) {
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

TEST(MatMulStressTest, RectangularOddShapes_Calculation) {
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
                // Because HELIX backend internally transposes B, we access B sequentially if it was B_T.
                // But in tests we access B directly as row-major.
                sum += a_ptr[i * k + l] * b_ptr[l * n + j];
            }
            EXPECT_NEAR(c_ptr[i * n + j], sum, 1e-4);
        }
    }
}

TEST(MatMulStressTest, RectangularOddShapes_KTail) {
    size_t m = 31;
    size_t k = 17;
    size_t n = 29;

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
