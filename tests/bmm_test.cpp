#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "grad_check.hpp"

using namespace helix;

// =============================================================================
// Forward Operation Tests
// =============================================================================

// 1. Basic Analytical Forward Correctness
TEST(BmmTest, BasicForwardAnalytical) {
    // 2 batches of (2x3) matrices
    Tensor a(
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,

         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f},
        Shape{2, 2, 3}
    );

    // 2 batches of (3x2) matrices
    Tensor b(
        {1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f,

         7.0f,
         8.0f,
         9.0f,
         10.0f,
         11.0f,
         12.0f},
        Shape{2, 3, 2}
    );

    Tensor c = a.bmm(b);

    EXPECT_EQ(c.shape(), Shape({2, 2, 2}));

    // Batch 0:
    // [0, 0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // [0, 1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    // [1, 0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
    // [1, 1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
    EXPECT_FLOAT_EQ(c.item({0, 0, 0}), 22.0f);
    EXPECT_FLOAT_EQ(c.item({0, 0, 1}), 28.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1, 0}), 49.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1, 1}), 64.0f);

    // Batch 1:
    // [0, 0] = 7*7 + 8*9 + 9*11 = 49 + 72 + 99 = 220
    // [0, 1] = 7*8 + 8*10 + 9*12 = 56 + 80 + 108 = 244
    // [1, 0] = 10*7 + 11*9 + 12*11 = 70 + 99 + 132 = 301
    // [1, 1] = 10*8 + 11*10 + 12*12 = 80 + 110 + 144 = 334
    EXPECT_FLOAT_EQ(c.item({1, 0, 0}), 220.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0, 1}), 244.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1, 0}), 301.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1, 1}), 334.0f);
}

// 2. Rank and Dimension Validation
TEST(BmmTest, DimensionAndRankValidation) {
    // Rank 1
    Tensor r1_a = Tensor::zeros({3});
    Tensor r1_b = Tensor::zeros({3});
    EXPECT_THROW((void)r1_a.bmm(r1_b), std::invalid_argument);

    // Rank 2 (should use matmul, bmm strictly rejects 2D)
    Tensor r2_a = Tensor::zeros({2, 3});
    Tensor r2_b = Tensor::zeros({3, 2});
    EXPECT_THROW((void)r2_a.bmm(r2_b), std::invalid_argument);

    // Rank 4
    Tensor r4_a = Tensor::zeros({1, 2, 3, 4});
    Tensor r4_b = Tensor::zeros({1, 2, 4, 3});
    EXPECT_THROW((void)r4_a.bmm(r4_b), std::invalid_argument);

    // Batch mismatch
    Tensor batch_a = Tensor::zeros({2, 3, 4});
    Tensor batch_b = Tensor::zeros({3, 4, 5});
    EXPECT_THROW((void)batch_a.bmm(batch_b), std::invalid_argument);

    // Inner dimension mismatch (4 != 5)
    Tensor inner_a = Tensor::zeros({2, 3, 4});
    Tensor inner_b = Tensor::zeros({2, 5, 2});
    EXPECT_THROW((void)inner_a.bmm(inner_b), std::invalid_argument);
}

// 3. Non-Contiguous Spatial Transposed Inputs
TEST(BmmTest, NonContiguousSpatialTransposed) {
    Tensor orig_a = Tensor::randn({2, 3, 2});
    Tensor orig_b = Tensor::randn({2, 2, 3});

    // Transpose matrix dimensions 1 and 2
    Tensor a = orig_a.transpose(1, 2);  // Shape [2, 2, 3], non-contiguous
    Tensor b = orig_b.transpose(1, 2);  // Shape [2, 3, 2], non-contiguous

    EXPECT_FALSE(a.is_contiguous());
    EXPECT_FALSE(b.is_contiguous());

    Tensor c = a.bmm(b);
    Tensor c_ref = a.clone().bmm(b.clone());

    EXPECT_EQ(c.shape(), Shape({2, 2, 2}));
    for (size_t b_idx = 0; b_idx < 2; ++b_idx) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_NEAR(c.item({b_idx, i, j}), c_ref.item({b_idx, i, j}), 1e-5f);
            }
        }
    }
}

// 4. Non-Zero Storage Offset Sliced Inputs
TEST(BmmTest, NonZeroStorageOffsetSliced) {
    Tensor large_a = Tensor::randn({4, 2, 3});
    Tensor large_b = Tensor::randn({4, 3, 2});

    // Slices on batch dimension create tensors with non-zero storage offsets
    Tensor a_slice = large_a.slice(0, 1, 3);  // Shape [2, 2, 3]
    Tensor b_slice = large_b.slice(0, 1, 3);  // Shape [2, 3, 2]

    EXPECT_TRUE(a_slice.is_shared());
    EXPECT_TRUE(b_slice.is_shared());

    Tensor c = a_slice.bmm(b_slice);
    Tensor c_ref = a_slice.clone().bmm(b_slice.clone());

    EXPECT_EQ(c.shape(), Shape({2, 2, 2}));
    for (size_t b_idx = 0; b_idx < 2; ++b_idx) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                EXPECT_NEAR(c.item({b_idx, i, j}), c_ref.item({b_idx, i, j}), 1e-5f);
            }
        }
    }
}

// 5. Multi-DType and Promotion Support
TEST(BmmTest, MultiDTypeSupport) {
    // Float64
    Tensor a_f64 = Tensor::ones({2, 2, 2}, DType::Float64);
    Tensor b_f64 = Tensor::ones({2, 2, 2}, DType::Float64);
    Tensor c_f64 = a_f64.bmm(b_f64);
    EXPECT_EQ(c_f64.dtype(), DType::Float64);
    EXPECT_DOUBLE_EQ(c_f64.data_ptr<double>()[0], 2.0);

    // Int32
    Tensor a_i32 = Tensor::ones({2, 2, 2}, DType::Int32);
    Tensor b_i32 = Tensor::ones({2, 2, 2}, DType::Int32);
    Tensor c_i32 = a_i32.bmm(b_i32);
    EXPECT_EQ(c_i32.dtype(), DType::Int32);
    EXPECT_EQ(c_i32.data_ptr<int32_t>()[0], 2);

    // Int64
    Tensor a_i64 = Tensor::ones({2, 2, 2}, DType::Int64);
    Tensor b_i64 = Tensor::ones({2, 2, 2}, DType::Int64);
    Tensor c_i64 = a_i64.bmm(b_i64);
    EXPECT_EQ(c_i64.dtype(), DType::Int64);
    EXPECT_EQ(c_i64.data_ptr<int64_t>()[0], 2);

    // Mixed Float32 * Float64 -> Float64
    Tensor a_f32 = Tensor::ones({2, 2, 2}, DType::Float32);
    Tensor c_mixed = a_f32.bmm(b_f64);
    EXPECT_EQ(c_mixed.dtype(), DType::Float64);
    EXPECT_DOUBLE_EQ(c_mixed.data_ptr<double>()[0], 2.0);
}

// 6. Zero-Extent Edge Cases (Forward)
TEST(BmmTest, ZeroExtentTensors) {
    // B = 0
    Tensor a_b0 = Tensor::zeros({0, 3, 4});
    Tensor b_b0 = Tensor::zeros({0, 4, 5});
    Tensor c_b0 = a_b0.bmm(b_b0);
    EXPECT_EQ(c_b0.shape(), Shape({0, 3, 5}));
    EXPECT_EQ(c_b0.numel(), 0);

    // M = 0
    Tensor a_m0 = Tensor::zeros({2, 0, 4});
    Tensor b_m0 = Tensor::zeros({2, 4, 5});
    Tensor c_m0 = a_m0.bmm(b_m0);
    EXPECT_EQ(c_m0.shape(), Shape({2, 0, 5}));
    EXPECT_EQ(c_m0.numel(), 0);

    // N = 0
    Tensor a_n0 = Tensor::zeros({2, 3, 4});
    Tensor b_n0 = Tensor::zeros({2, 4, 0});
    Tensor c_n0 = a_n0.bmm(b_n0);
    EXPECT_EQ(c_n0.shape(), Shape({2, 3, 0}));
    EXPECT_EQ(c_n0.numel(), 0);

    // K = 0 (empty sum => zeros)
    Tensor a_k0 = Tensor::zeros({2, 3, 0});
    Tensor b_k0 = Tensor::zeros({2, 0, 4});
    Tensor c_k0 = a_k0.bmm(b_k0);
    EXPECT_EQ(c_k0.shape(), Shape({2, 3, 4}));
    EXPECT_EQ(c_k0.numel(), 24);
    for (size_t i = 0; i < c_k0.numel(); ++i) {
        EXPECT_FLOAT_EQ(c_k0.data_ptr<float>()[i], 0.0f);
    }
}

// 7. Functional Parity and Manual Slice MatMul Parity
TEST(BmmTest, FunctionalAndManualParity) {
    Tensor a = Tensor::randn({4, 3, 5});
    Tensor b = Tensor::randn({4, 5, 2});

    // Functional API parity
    Tensor c_member = a.bmm(b);
    Tensor c_free = helix::bmm(a, b);
    EXPECT_EQ(c_member.shape(), c_free.shape());
    for (size_t i = 0; i < c_member.numel(); ++i) {
        EXPECT_FLOAT_EQ(c_member.data_ptr<float>()[i], c_free.data_ptr<float>()[i]);
    }

    // Manual slice-and-matmul parity
    for (size_t i = 0; i < 4; ++i) {
        Tensor a_i = a.slice(0, i, i + 1).squeeze(0);
        Tensor b_i = b.slice(0, i, i + 1).squeeze(0);
        Tensor c_i = a_i.matmul(b_i);
        Tensor expected_i = c_member.slice(0, i, i + 1).squeeze(0);

        for (size_t r = 0; r < 3; ++r) {
            for (size_t col = 0; col < 2; ++col) {
                EXPECT_NEAR(c_i.item({r, col}), expected_i.item({r, col}), 1e-5f);
            }
        }
    }
}

// =============================================================================
// Autograd Tests
// =============================================================================

class BmmAutogradTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

// 8. Zero-Extent Backward for K = 0
TEST_F(BmmAutogradTest, ZeroExtentBackwardKZero) {
    Tensor a = Tensor::zeros({2, 3, 0});
    Tensor b = Tensor::zeros({2, 0, 4});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor loss = a.bmm(b).sum();
    loss.backward();

    EXPECT_EQ(a.grad().shape(), Shape({2, 3, 0}));
    EXPECT_EQ(b.grad().shape(), Shape({2, 0, 4}));
    EXPECT_EQ(a.grad().numel(), 0);
    EXPECT_EQ(b.grad().numel(), 0);
}

// 9. Analytical Autograd Backward Check
TEST_F(BmmAutogradTest, AnalyticalBackward) {
    // Inputs [2, 2, 2]
    Tensor a(
        {1.0f,
         2.0f,
         3.0f,
         4.0f,

         5.0f,
         6.0f,
         7.0f,
         8.0f},
        Shape{2, 2, 2}
    );

    Tensor b(
        {2.0f,
         0.0f,
         1.0f,
         3.0f,

         1.0f,
         1.0f,
         2.0f,
         0.0f},
        Shape{2, 2, 2}
    );

    Tensor w(
        {1.0f,
         0.5f,
         2.0f,
         1.0f,

         0.5f,
         1.0f,
         1.0f,
         2.0f},
        Shape{2, 2, 2}
    );

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor loss = (a.bmm(b) * w).sum();
    loss.backward();

    // Derived analytical gradients:
    // dA_0 = W_0 * B_0^T = [[2, 2.5], [4, 5]]
    // dA_1 = W_1 * B_1^T = [[1.5, 1], [3, 2]]
    EXPECT_FLOAT_EQ(a.grad().item({0, 0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.grad().item({0, 0, 1}), 2.5f);
    EXPECT_FLOAT_EQ(a.grad().item({0, 1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(a.grad().item({0, 1, 1}), 5.0f);

    EXPECT_FLOAT_EQ(a.grad().item({1, 0, 0}), 1.5f);
    EXPECT_FLOAT_EQ(a.grad().item({1, 0, 1}), 1.0f);
    EXPECT_FLOAT_EQ(a.grad().item({1, 1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(a.grad().item({1, 1, 1}), 2.0f);

    // dB_0 = A_0^T * W_0 = [[7, 3.5], [10, 5]]
    // dB_1 = A_1^T * W_1 = [[9.5, 19], [11, 22]]
    EXPECT_FLOAT_EQ(b.grad().item({0, 0, 0}), 7.0f);
    EXPECT_FLOAT_EQ(b.grad().item({0, 0, 1}), 3.5f);
    EXPECT_FLOAT_EQ(b.grad().item({0, 1, 0}), 10.0f);
    EXPECT_FLOAT_EQ(b.grad().item({0, 1, 1}), 5.0f);

    EXPECT_FLOAT_EQ(b.grad().item({1, 0, 0}), 9.5f);
    EXPECT_FLOAT_EQ(b.grad().item({1, 0, 1}), 19.0f);
    EXPECT_FLOAT_EQ(b.grad().item({1, 1, 0}), 11.0f);
    EXPECT_FLOAT_EQ(b.grad().item({1, 1, 1}), 22.0f);
}

// 10. Autograd Single Gradient Branch
TEST_F(BmmAutogradTest, SingleGradBranch) {
    // Only A requires grad
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(false);

        Tensor loss = a.bmm(b).sum();
        loss.backward();

        EXPECT_TRUE(a.has_grad());
        EXPECT_FALSE(b.requires_grad());
        EXPECT_FALSE(b.has_grad());
    }

    // Only B requires grad
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(false);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        loss.backward();

        EXPECT_FALSE(a.requires_grad());
        EXPECT_FALSE(a.has_grad());
        EXPECT_TRUE(b.has_grad());
    }
}

// 11. Numerical Grad Check (Deterministic Secondary Check)
TEST_F(BmmAutogradTest, NumericalGradCheck) {
    // Deterministic fixed values in [-0.8, 0.8] without randn
    std::vector<float> a_data(24);
    for (size_t i = 0; i < 24; ++i) {
        a_data[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.2f;
    }
    Tensor a(a_data, Shape{2, 3, 4});

    std::vector<float> b_data(16);
    for (size_t i = 0; i < 16; ++i) {
        b_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    }
    Tensor b(b_data, Shape{2, 4, 2});

    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].bmm(inputs[1]).sum(); };

    EXPECT_TRUE(gradient_check(func, {a, b}));
}

// 12. Zero-Extent Backward for Outer Dimensions (B=0, M=0, N=0)
TEST_F(BmmAutogradTest, ZeroExtentBackwardOuterDims) {
    // B = 0
    {
        Tensor a = Tensor::zeros({0, 3, 4});
        Tensor b = Tensor::zeros({0, 4, 5});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        loss.backward();

        EXPECT_EQ(a.grad().shape(), Shape({0, 3, 4}));
        EXPECT_EQ(b.grad().shape(), Shape({0, 4, 5}));
        EXPECT_EQ(a.grad().numel(), 0);
        EXPECT_EQ(b.grad().numel(), 0);
    }

    // M = 0
    {
        Tensor a = Tensor::zeros({2, 0, 4});
        Tensor b = Tensor::zeros({2, 4, 5});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        loss.backward();

        EXPECT_EQ(a.grad().shape(), Shape({2, 0, 4}));
        EXPECT_EQ(b.grad().shape(), Shape({2, 4, 5}));
        EXPECT_EQ(a.grad().numel(), 0);
        EXPECT_EQ(b.grad().numel(), 40);
        for (size_t i = 0; i < b.grad().numel(); ++i) {
            EXPECT_FLOAT_EQ(b.grad().data_ptr<float>()[i], 0.0f);
        }
    }

    // N = 0
    {
        Tensor a = Tensor::zeros({2, 3, 4});
        Tensor b = Tensor::zeros({2, 4, 0});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        loss.backward();

        EXPECT_EQ(a.grad().shape(), Shape({2, 3, 4}));
        EXPECT_EQ(b.grad().shape(), Shape({2, 4, 0}));
        EXPECT_EQ(a.grad().numel(), 24);
        EXPECT_EQ(b.grad().numel(), 0);
        for (size_t i = 0; i < a.grad().numel(); ++i) {
            EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[i], 0.0f);
        }
    }
}

// 13. Mixed DType Autograd Backward (Float32 x Float64)
TEST_F(BmmAutogradTest, MixedDTypeBackward) {
    Tensor a = Tensor::ones({2, 2, 2}, DType::Float32);
    Tensor b = Tensor::ones({2, 2, 2}, DType::Float64);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor loss = a.bmm(b).sum();
    loss.backward();

    EXPECT_EQ(a.grad().dtype(), DType::Float32);
    EXPECT_EQ(b.grad().dtype(), DType::Float64);
    EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[0], 2.0f);
    EXPECT_DOUBLE_EQ(b.grad().data_ptr<double>()[0], 2.0);
}

// 14. In-place Version Tracking Error with no_grad Mutation Guard
TEST_F(BmmAutogradTest, InplaceVersionTrackingError) {
    // Case A: A modified in-place under no_grad
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        {
            no_grad guard;
            a.add_(Tensor::ones(a.shape()));
        }
        EXPECT_THROW(loss.backward(), std::runtime_error);
    }

    // Case B: B modified in-place under no_grad
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        {
            no_grad guard;
            b.add_(Tensor::ones(b.shape()));
        }
        EXPECT_THROW(loss.backward(), std::runtime_error);
    }

    // Case C: Non-grad operand B modified under no_grad
    // (B is still saved by BmmBackward to compute dL/dA = grad_out.bmm(B^T))
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(false);

        Tensor loss = a.bmm(b).sum();
        {
            no_grad guard;
            b.add_(Tensor::ones(b.shape()));
        }
        EXPECT_THROW(loss.backward(), std::runtime_error);
    }
}

// 15. Gradient Accumulation Across Multiple Backwards
TEST_F(BmmAutogradTest, GradientAccumulation) {
    // Multi-Forward Independent Graphs
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss1 = a.bmm(b).sum();
        loss1.backward();
        Tensor grad_a_1 = a.grad().clone();
        Tensor grad_b_1 = b.grad().clone();

        Tensor loss2 = a.bmm(b).sum();
        loss2.backward();

        for (size_t i = 0; i < a.numel(); ++i) {
            EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[i], grad_a_1.data_ptr<float>()[i] * 2.0f);
        }
        for (size_t i = 0; i < b.numel(); ++i) {
            EXPECT_FLOAT_EQ(b.grad().data_ptr<float>()[i], grad_b_1.data_ptr<float>()[i] * 2.0f);
        }
    }

    // Retained Graph Backward
    {
        Tensor a = Tensor::randn({2, 2, 2});
        Tensor b = Tensor::randn({2, 2, 2});
        a.set_requires_grad(true);
        b.set_requires_grad(true);

        Tensor loss = a.bmm(b).sum();
        loss.backward({}, true);
        Tensor grad_a_1 = a.grad().clone();
        Tensor grad_b_1 = b.grad().clone();

        loss.backward();

        for (size_t i = 0; i < a.numel(); ++i) {
            EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[i], grad_a_1.data_ptr<float>()[i] * 2.0f);
        }
        for (size_t i = 0; i < b.numel(); ++i) {
            EXPECT_FLOAT_EQ(b.grad().data_ptr<float>()[i], grad_b_1.data_ptr<float>()[i] * 2.0f);
        }
    }
}
