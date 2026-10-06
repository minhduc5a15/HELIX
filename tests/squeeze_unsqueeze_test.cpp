#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "autograd/engine.hpp"
#include "core/math_utils.hpp"
#include "core/tensor.hpp"

using namespace helix;

// =============================================================================
// Helper Function Tests
// =============================================================================

TEST(CheckedStrideMulTest, CorrectCalculationsAndOverflow) {
    EXPECT_EQ(checked_stride_mul(12, 4), 48);
    EXPECT_EQ(checked_stride_mul(12, 0), 0);
    EXPECT_EQ(checked_stride_mul(0, 10), 0);
    EXPECT_EQ(checked_stride_mul(-4, 3), -12);
    EXPECT_EQ(checked_stride_mul(-4, 0), 0);

    const ptrdiff_t max_val = std::numeric_limits<ptrdiff_t>::max();
    EXPECT_THROW(checked_stride_mul(max_val, 2), std::overflow_error);
    EXPECT_THROW(checked_stride_mul(-max_val, 2), std::overflow_error);
}

// =============================================================================
// Tensor Unsqueeze Tests
// =============================================================================

TEST(TensorUnsqueezeTest, ContiguousRankExpansions) {
    Tensor x = Tensor::zeros({2, 3, 4});
    ASSERT_EQ(x.stride().vec(), (std::vector<ptrdiff_t>{12, 4, 1}));

    // unsqueeze(0) -> [1, 2, 3, 4], stride [24, 12, 4, 1]
    Tensor u0 = x.unsqueeze(0);
    EXPECT_EQ(u0.shape().vec(), (std::vector<size_t>{1, 2, 3, 4}));
    EXPECT_EQ(u0.stride().vec(), (std::vector<ptrdiff_t>{24, 12, 4, 1}));

    // unsqueeze(1) -> [2, 1, 3, 4], stride [12, 12, 4, 1]
    Tensor u1 = x.unsqueeze(1);
    EXPECT_EQ(u1.shape().vec(), (std::vector<size_t>{2, 1, 3, 4}));
    EXPECT_EQ(u1.stride().vec(), (std::vector<ptrdiff_t>{12, 12, 4, 1}));

    // unsqueeze(2) -> [2, 3, 1, 4], stride [12, 4, 4, 1]
    Tensor u2 = x.unsqueeze(2);
    EXPECT_EQ(u2.shape().vec(), (std::vector<size_t>{2, 3, 1, 4}));
    EXPECT_EQ(u2.stride().vec(), (std::vector<ptrdiff_t>{12, 4, 4, 1}));

    // unsqueeze(3) -> [2, 3, 4, 1], stride [12, 4, 1, 1]
    Tensor u3 = x.unsqueeze(3);
    EXPECT_EQ(u3.shape().vec(), (std::vector<size_t>{2, 3, 4, 1}));
    EXPECT_EQ(u3.stride().vec(), (std::vector<ptrdiff_t>{12, 4, 1, 1}));
}

TEST(TensorUnsqueezeTest, NegativeIndexing) {
    Tensor x = Tensor::zeros({2, 3, 4});

    Tensor u_neg1 = x.unsqueeze(-1);  // norm_dim = 3
    EXPECT_EQ(u_neg1.shape().vec(), (std::vector<size_t>{2, 3, 4, 1}));
    EXPECT_EQ(u_neg1.stride().vec(), (std::vector<ptrdiff_t>{12, 4, 1, 1}));

    Tensor u_neg2 = x.unsqueeze(-2);  // norm_dim = 2
    EXPECT_EQ(u_neg2.shape().vec(), (std::vector<size_t>{2, 3, 1, 4}));
    EXPECT_EQ(u_neg2.stride().vec(), (std::vector<ptrdiff_t>{12, 4, 4, 1}));

    Tensor u_neg4 = x.unsqueeze(-4);  // norm_dim = 0
    EXPECT_EQ(u_neg4.shape().vec(), (std::vector<size_t>{1, 2, 3, 4}));
    EXPECT_EQ(u_neg4.stride().vec(), (std::vector<ptrdiff_t>{24, 12, 4, 1}));
}

TEST(TensorUnsqueezeTest, ScalarTensor) {
    Tensor s(Shape{}, DType::Float32);
    s.data_ptr<float>()[0] = 42.0f;
    EXPECT_EQ(s.rank(), 0);
    EXPECT_TRUE(s.shape().empty());

    Tensor u0 = s.unsqueeze(0);
    EXPECT_EQ(u0.shape().vec(), (std::vector<size_t>{1}));
    EXPECT_EQ(u0.stride().vec(), (std::vector<ptrdiff_t>{1}));
    EXPECT_FLOAT_EQ(u0.data_ptr<float>()[0], 42.0f);

    Tensor u_neg1 = s.unsqueeze(-1);
    EXPECT_EQ(u_neg1.shape().vec(), (std::vector<size_t>{1}));
    EXPECT_EQ(u_neg1.stride().vec(), (std::vector<ptrdiff_t>{1}));

    EXPECT_THROW((void)s.unsqueeze(1), std::out_of_range);
    EXPECT_THROW((void)s.unsqueeze(-2), std::out_of_range);
}

TEST(TensorUnsqueezeTest, ZeroExtentTensor) {
    Tensor z = Tensor::zeros({0, 5});
    EXPECT_EQ(z.shape().vec(), (std::vector<size_t>{0, 5}));

    Tensor u0 = z.unsqueeze(0);
    EXPECT_EQ(u0.shape().vec(), (std::vector<size_t>{1, 0, 5}));

    Tensor u1 = z.unsqueeze(1);
    EXPECT_EQ(u1.shape().vec(), (std::vector<size_t>{0, 1, 5}));

    Tensor u2 = z.unsqueeze(2);
    EXPECT_EQ(u2.shape().vec(), (std::vector<size_t>{0, 5, 1}));
}

TEST(TensorUnsqueezeTest, OutOfRangeThrows) {
    Tensor x = Tensor::zeros({2, 3, 4});  // rank = 3
    EXPECT_THROW((void)x.unsqueeze(4), std::out_of_range);
    EXPECT_THROW((void)x.unsqueeze(-5), std::out_of_range);
}

TEST(TensorUnsqueezeTest, NonContiguousAliasingAndStride) {
    // Base shape [2, 3, 4] with sequential values
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    Tensor base(data, {2, 3, 4});

    // Transpose dims 0 and 1: shape [3, 2, 4], non-contiguous stride [4, 12, 1]
    Tensor x = base.transpose(0, 1);
    ASSERT_FALSE(x.is_contiguous());
    ASSERT_EQ(x.shape().vec(), (std::vector<size_t>{3, 2, 4}));
    ASSERT_EQ(x.stride().vec(), (std::vector<ptrdiff_t>{4, 12, 1}));

    // Unsqueeze at dim 1: shape [3, 1, 2, 4], stride [4, 24, 12, 1]
    Tensor y = x.unsqueeze(1);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{3, 1, 2, 4}));
    EXPECT_EQ(y.stride().vec(), (std::vector<ptrdiff_t>{4, 24, 12, 1}));

    // Zero-copy storage and pointer aliasing invariants
    EXPECT_EQ(y.impl()->storage(), x.impl()->storage());
    EXPECT_EQ(y.impl()->storage_offset(), x.impl()->storage_offset());
    EXPECT_EQ(y.data_ptr<float>(), x.data_ptr<float>());

    // Verify in-place write through aliased view propagates to base and x
    y.data_ptr<float>()[0] = 999.0f;
    EXPECT_FLOAT_EQ(base.data_ptr<float>()[0], 999.0f);
    EXPECT_FLOAT_EQ(x.data_ptr<float>()[0], 999.0f);
}

// =============================================================================
// Tensor Squeeze Tests
// =============================================================================

TEST(TensorSqueezeTest, DefaultAllSingletonRemoval) {
    Tensor x = Tensor::zeros({1, 2, 1, 3, 1});
    Tensor y = x.squeeze();
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(y.impl()->storage(), x.impl()->storage());
    EXPECT_EQ(y.impl()->storage_offset(), x.impl()->storage_offset());
}

TEST(TensorSqueezeTest, AllSingletonToScalar) {
    Tensor x = Tensor::zeros({1, 1, 1});
    Tensor y = x.squeeze();
    EXPECT_EQ(y.rank(), 0);
    EXPECT_TRUE(y.shape().empty());
    EXPECT_EQ(y.impl()->storage(), x.impl()->storage());
}

TEST(TensorSqueezeTest, NoSingletonReturnsSelf) {
    Tensor x = Tensor::zeros({2, 3});
    Tensor y = x.squeeze();
    EXPECT_EQ(y.impl(), x.impl());

    // Invariant: no autograd node inserted when no singleton exists
    init_autograd();
    Tensor x_grad = Tensor::zeros({2, 3});
    x_grad.set_requires_grad(true);
    Tensor y_grad = x_grad.squeeze();
    EXPECT_EQ(y_grad.impl(), x_grad.impl());
}

TEST(TensorSqueezeTest, TargetedSingletonRemoval) {
    Tensor x = Tensor::zeros({2, 1, 3});
    Tensor y = x.squeeze(1);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(y.impl()->storage(), x.impl()->storage());
}

TEST(TensorSqueezeTest, TargetedNonSingletonNoOp) {
    Tensor x = Tensor::zeros({2, 4});
    Tensor y = x.squeeze(1);
    EXPECT_EQ(y.impl(), x.impl());

    // Invariant: no autograd node inserted on non-singleton target
    init_autograd();
    Tensor x_grad = Tensor::zeros({2, 4});
    x_grad.set_requires_grad(true);
    Tensor y_grad = x_grad.squeeze(1);
    EXPECT_EQ(y_grad.impl(), x_grad.impl());
}

TEST(TensorSqueezeTest, ZeroExtentNotConfusedWithSingleton) {
    Tensor x = Tensor::zeros({0, 1, 5});

    Tensor y_all = x.squeeze();
    EXPECT_EQ(y_all.shape().vec(), (std::vector<size_t>{0, 5}));

    Tensor y_dim1 = x.squeeze(1);
    EXPECT_EQ(y_dim1.shape().vec(), (std::vector<size_t>{0, 5}));

    Tensor y_dim0 = x.squeeze(0);  // shape[0] == 0 != 1 -> no-op
    EXPECT_EQ(y_dim0.impl(), x.impl());
}

TEST(TensorSqueezeTest, NegativeIndexing) {
    Tensor x = Tensor::zeros({2, 1, 3});
    Tensor y = x.squeeze(-2);  // norm_dim = 1
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{2, 3}));
}

TEST(TensorSqueezeTest, ScalarNoOp) {
    Tensor s(Shape{}, DType::Float32);
    s.data_ptr<float>()[0] = 10.0f;
    EXPECT_EQ(s.squeeze().impl(), s.impl());
    EXPECT_EQ(s.squeeze(0).impl(), s.impl());
    EXPECT_EQ(s.squeeze(-1).impl(), s.impl());

    EXPECT_THROW((void)s.squeeze(1), std::out_of_range);
    EXPECT_THROW((void)s.squeeze(-2), std::out_of_range);
}

TEST(TensorSqueezeTest, OutOfRangeThrows) {
    Tensor x = Tensor::zeros({2, 3});  // rank = 2
    EXPECT_THROW((void)x.squeeze(2), std::out_of_range);
    EXPECT_THROW((void)x.squeeze(-3), std::out_of_range);
}

TEST(TensorSqueezeTest, NonContiguousAliasingAndStride) {
    // Create base [2, 3] with sequential data, transpose to [3, 2] (stride [1, 3])
    std::vector<float> data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor base(data, {2, 3});
    Tensor t = base.transpose(0, 1);  // shape [3, 2], stride [1, 3]

    // Unsqueeze at 1 produces [3, 1, 2], stride [1, 6, 3]
    Tensor x = t.unsqueeze(1);
    ASSERT_EQ(x.shape().vec(), (std::vector<size_t>{3, 1, 2}));
    ASSERT_EQ(x.stride().vec(), (std::vector<ptrdiff_t>{1, 6, 3}));

    // Squeeze at 1 drops singleton axis 1: shape [3, 2], stride [1, 3]
    Tensor y = x.squeeze(1);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(y.stride().vec(), (std::vector<ptrdiff_t>{1, 3}));

    // Zero-copy storage and pointer aliasing invariants
    EXPECT_EQ(y.impl()->storage(), x.impl()->storage());
    EXPECT_EQ(y.impl()->storage_offset(), x.impl()->storage_offset());
    EXPECT_EQ(y.data_ptr<float>(), x.data_ptr<float>());
}

// =============================================================================
// Deterministic Autograd Tests (Weighted Analytical Backward)
// =============================================================================

class SqueezeUnsqueezeAutogradTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(SqueezeUnsqueezeAutogradTest, UnsqueezeBackwardWeighted) {
    // Isolated unsqueeze on non-contiguous tensor
    // Base shape [2, 3] with requires_grad = true
    std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x(x_data, {2, 3});
    x.set_requires_grad(true);

    // Transpose -> [3, 2] non-contiguous
    Tensor t = x.transpose(0, 1);

    // Unsqueeze at 1 -> [3, 1, 2]
    Tensor u = t.unsqueeze(1);

    // Create unique deterministic weights W of shape [3, 1, 2]
    // w[j, 0, i] = (j + 1) * 10 + (i + 1)
    std::vector<float> w_data(6);
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            w_data[j * 2 + i] = static_cast<float>((j + 1) * 10 + (i + 1));
        }
    }
    Tensor w(w_data, {3, 1, 2});

    // Loss = sum(u * w)
    Tensor loss = (u * w).sum();
    loss.backward();

    // Mathematically:
    // x[i, j] = t[j, i] = u[j, 0, i]
    // Loss = sum_{j, i} u[j, 0, i] * w[j, 0, i] = sum_{i, j} x[i, j] * w[j, 0, i]
    // Therefore: dLoss / dx[i, j] = w[j, 0, i]
    Tensor grad = x.grad();
    ASSERT_EQ(grad.shape().vec(), (std::vector<size_t>{2, 3}));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float expected_grad = static_cast<float>((j + 1) * 10 + (i + 1));
            const float actual_grad = grad.data_ptr<float>()[i * 3 + j];
            EXPECT_NEAR(actual_grad, expected_grad, 1e-5f);
        }
    }
}

TEST_F(SqueezeUnsqueezeAutogradTest, SqueezeBackwardWeighted) {
    // Isolated squeeze on non-contiguous tensor
    // Base shape [2, 1, 3] with requires_grad = true
    std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x(x_data, {2, 1, 3});
    x.set_requires_grad(true);

    // Transpose dims 0 and 2 -> shape [3, 1, 2] non-contiguous
    Tensor t = x.transpose(0, 2);

    // Squeeze at 1 -> [3, 2]
    Tensor s = t.squeeze(1);

    // Create unique deterministic weights W of shape [3, 2]
    // w[j, i] = (j + 1) * 10 + (i + 1)
    std::vector<float> w_data(6);
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            w_data[j * 2 + i] = static_cast<float>((j + 1) * 10 + (i + 1));
        }
    }
    Tensor w(w_data, {3, 2});

    // Loss = sum(s * w)
    Tensor loss = (s * w).sum();
    loss.backward();

    // Mathematically:
    // x[i, 0, j] = t[j, 0, i] = s[j, i]
    // Loss = sum_{j, i} s[j, i] * w[j, i] = sum_{i, j} x[i, 0, j] * w[j, i]
    // Therefore: dLoss / dx[i, 0, j] = w[j, i]
    Tensor grad = x.grad();
    ASSERT_EQ(grad.shape().vec(), (std::vector<size_t>{2, 1, 3}));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float expected_grad = static_cast<float>((j + 1) * 10 + (i + 1));
            const float actual_grad = grad.data_ptr<float>()[i * 3 + j];
            EXPECT_NEAR(actual_grad, expected_grad, 1e-5f);
        }
    }
}

TEST_F(SqueezeUnsqueezeAutogradTest, ChainedTransposeUnsqueezeSqueezeBackward) {
    // Chained: transpose -> unsqueeze -> squeeze -> mul(w) -> sum -> backward
    std::vector<float> x_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x(x_data, {2, 3});
    x.set_requires_grad(true);

    Tensor t = x.transpose(0, 1);  // [3, 2]
    Tensor u = t.unsqueeze(1);     // [3, 1, 2]
    Tensor s = u.squeeze(1);       // [3, 2]

    std::vector<float> w_data(6);
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            w_data[j * 2 + i] = static_cast<float>((j + 1) * 100 + (i + 1));
        }
    }
    Tensor w(w_data, {3, 2});

    Tensor loss = (s * w).sum();
    loss.backward();

    Tensor grad = x.grad();
    ASSERT_EQ(grad.shape().vec(), (std::vector<size_t>{2, 3}));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const float expected_grad = static_cast<float>((j + 1) * 100 + (i + 1));
            const float actual_grad = grad.data_ptr<float>()[i * 3 + j];
            EXPECT_NEAR(actual_grad, expected_grad, 1e-5f);
        }
    }
}

// =============================================================================
// Functional Parity Tests
// =============================================================================

TEST(FunctionalParityTest, FreeFunctionsMatchMemberMethods) {
    Tensor x = Tensor::zeros({2, 1, 3});

    // unsqueeze
    Tensor u_mem = x.unsqueeze(0);
    Tensor u_free = helix::unsqueeze(x, 0);
    EXPECT_EQ(u_mem.shape(), u_free.shape());
    EXPECT_EQ(u_mem.stride(), u_free.stride());

    // squeeze(dim)
    Tensor sq_dim_mem = x.squeeze(1);
    Tensor sq_dim_free = helix::squeeze(x, 1);
    EXPECT_EQ(sq_dim_mem.shape(), sq_dim_free.shape());
    EXPECT_EQ(sq_dim_mem.stride(), sq_dim_free.stride());

    // squeeze()
    Tensor sq_mem = x.squeeze();
    Tensor sq_free = helix::squeeze(x);
    EXPECT_EQ(sq_mem.shape(), sq_free.shape());
    EXPECT_EQ(sq_mem.stride(), sq_free.stride());
}
