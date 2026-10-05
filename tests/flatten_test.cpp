#include <gtest/gtest.h>

#include <vector>

#include "grad_check.hpp"
#include "helix.hpp"

using namespace helix;

// =============================================================================
// Suite 1: TensorFlatten (Core Tensor & Functional API)
// =============================================================================

TEST(TensorFlatten, DefaultFullFlatten) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});
    Tensor flat = a.flatten();
    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shape()[0], 6);
    EXPECT_TRUE(flat.is_contiguous());
    EXPECT_FLOAT_EQ(flat.item({0}), 1.0f);
    EXPECT_FLOAT_EQ(flat.item({5}), 6.0f);
}

TEST(TensorFlatten, FunctionalAPIParity) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor f1 = a.flatten();
    Tensor f2 = flatten(a);
    EXPECT_EQ(f1.shape(), f2.shape());
    EXPECT_FLOAT_EQ(f1.item({3}), f2.item({3}));
}

TEST(TensorFlatten, CustomRangeAndNegativeIndexing) {
    Tensor x = Tensor::zeros({2, 3, 4, 5});
    EXPECT_EQ(x.flatten(1, 2).shape().vec(), (std::vector<size_t>{2, 12, 5}));
    EXPECT_EQ(x.flatten(-2, -1).shape().vec(), (std::vector<size_t>{2, 3, 20}));
    EXPECT_EQ(x.flatten(1, -1).shape().vec(), (std::vector<size_t>{2, 60}));
    EXPECT_EQ(x.flatten(0, -1).shape().vec(), (std::vector<size_t>{120}));
}

TEST(TensorFlatten, IdentitySameDimReturnsSelfOnNonContiguous) {
    // Tests that start_dim == end_dim returns *this directly without calling reshape(shape())
    // which would otherwise needlessly clone a non-contiguous tensor.
    Tensor base = Tensor::randn({2, 3, 4});
    Tensor transposed = base.transpose(0, 1);  // shape {3, 2, 4}, non-contiguous
    EXPECT_FALSE(transposed.is_contiguous());

    const Stride orig_stride = transposed.stride();
    const auto orig_impl = transposed.impl();

    Tensor flat = transposed.flatten(1, 1);

    EXPECT_EQ(flat.impl(), orig_impl);
    EXPECT_FALSE(flat.is_contiguous());
    EXPECT_EQ(flat.stride(), orig_stride);
    EXPECT_EQ(flat.shape().vec(), (std::vector<size_t>{3, 2, 4}));
}

TEST(TensorFlatten, ScalarTensor) {
    Tensor scalar(Shape{}, DType::Float32);
    scalar.data_ptr<float>()[0] = 3.14f;

    Tensor flat = scalar.flatten(0, -1);
    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shape()[0], 1);
    EXPECT_FLOAT_EQ(flat.item({0}), 3.14f);

    EXPECT_THROW((void)scalar.flatten(1, -1), std::out_of_range);
}

TEST(TensorFlatten, ZeroExtentDimensions) {
    Tensor empty2d = Tensor::empty({0, 5});
    Tensor flat1 = empty2d.flatten(0, -1);
    EXPECT_EQ(flat1.shape().vec(), (std::vector<size_t>{0}));

    Tensor empty3d = Tensor::empty({2, 0, 5});
    Tensor flat2 = empty3d.flatten(1, -1);
    EXPECT_EQ(flat2.shape().vec(), (std::vector<size_t>{2, 0}));
}

TEST(TensorFlatten, InvalidDimensionsThrow) {
    Tensor x = Tensor::zeros({2, 3, 4});
    EXPECT_THROW((void)x.flatten(2, 1), std::invalid_argument);
    EXPECT_THROW((void)x.flatten(-1, -2), std::invalid_argument);
    EXPECT_THROW((void)x.flatten(0, 3), std::out_of_range);
    EXPECT_THROW((void)x.flatten(-4, 1), std::out_of_range);
}

TEST(TensorFlatten, NonContiguousFlattenDeterministicDataOrder) {
    // 2x3x4 tensor with deterministic values: val(i, j, k) = i*12 + j*4 + k
    std::vector<float> data(24);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                data[i * 12 + j * 4 + k] = static_cast<float>(i * 12 + j * 4 + k);
            }
        }
    }
    Tensor base(data, Shape{2, 3, 4});
    Tensor transposed = base.transpose(0, 1);  // shape {3, 2, 4}
    EXPECT_FALSE(transposed.is_contiguous());

    // Flatten dims 1 and 2: resulting shape {3, 8}
    Tensor flat = transposed.flatten(1, -1);
    EXPECT_TRUE(flat.is_contiguous());
    EXPECT_EQ(flat.shape().vec(), (std::vector<size_t>{3, 8}));

    // Verify logical ordering: flat[j, i * 4 + k] == base[i, j, k]
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            for (size_t k = 0; k < 4; ++k) {
                const float expected = static_cast<float>(i * 12 + j * 4 + k);
                EXPECT_FLOAT_EQ(flat.item({j, i * 4 + k}), expected);
            }
        }
    }
}

TEST(TensorFlatten, DTypePreservation) {
    Tensor int_tensor = Tensor::ones({2, 3, 4}, DType::Int64);
    Tensor flat_int = int_tensor.flatten(1, -1);
    EXPECT_EQ(flat_int.dtype(), DType::Int64);
    EXPECT_EQ(flat_int.shape().vec(), (std::vector<size_t>{2, 12}));

    Tensor float64_tensor = Tensor::ones({2, 3, 4}, DType::Float64);
    Tensor flat_f64 = float64_tensor.flatten(1, -1);
    EXPECT_EQ(flat_f64.dtype(), DType::Float64);
    EXPECT_EQ(flat_f64.shape().vec(), (std::vector<size_t>{2, 12}));
}

TEST(TensorFlatten, SliceDerivedNonContiguous) {
    // 2x4x3 tensor
    std::vector<float> data(24);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    Tensor base(data, Shape{2, 4, 3});
    // Slice dimension 1 from index 1 to 3 -> shape {2, 2, 3}
    Tensor sliced = base.slice(1, 1, 3);
    EXPECT_FALSE(sliced.is_contiguous());

    Tensor flat = sliced.flatten(1, -1);
    EXPECT_TRUE(flat.is_contiguous());
    EXPECT_EQ(flat.shape().vec(), (std::vector<size_t>{2, 6}));

    // Check data integrity:
    // sliced[0, 0, k] = base[0, 1, k] -> data[3 + k]
    // sliced[0, 1, k] = base[0, 2, k] -> data[6 + k]
    // sliced[1, 0, k] = base[1, 1, k] -> data[12 + 3 + k]
    // sliced[1, 1, k] = base[1, 2, k] -> data[12 + 6 + k]
    for (size_t k = 0; k < 3; ++k) {
        EXPECT_FLOAT_EQ(flat.item({0, k}), static_cast<float>(3 + k));
        EXPECT_FLOAT_EQ(flat.item({0, 3 + k}), static_cast<float>(6 + k));
        EXPECT_FLOAT_EQ(flat.item({1, k}), static_cast<float>(15 + k));
        EXPECT_FLOAT_EQ(flat.item({1, 3 + k}), static_cast<float>(18 + k));
    }
}

// =============================================================================
// Suite 2: FlattenModule (Neural Network Layer & Autograd)
// =============================================================================

class FlattenModuleTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(FlattenModuleTest, DefaultForwardPreservesBatch) {
    Flatten flatten_layer;
    EXPECT_EQ(flatten_layer.start_dim(), 1);
    EXPECT_EQ(flatten_layer.end_dim(), -1);

    Tensor input = Tensor::randn({16, 3, 32, 32});
    Tensor output = flatten_layer.forward(input);
    EXPECT_EQ(output.shape().vec(), (std::vector<size_t>{16, 3072}));
}

TEST_F(FlattenModuleTest, SequentialIntegrationAndBackwardToInput) {
    Sequential model(Flatten(), Linear(12, 2));
    Tensor input = Tensor::randn({2, 3, 4});
    input.set_requires_grad(true);

    Tensor output = model.forward(input);
    EXPECT_EQ(output.shape().vec(), (std::vector<size_t>{2, 2}));

    Tensor loss = output.sum();
    loss.backward();

    // Check gradient propagated back to input
    ASSERT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad().shape(), input.shape());

    // Check model parameters received gradient
    auto params = model.parameters();
    ASSERT_EQ(params.size(), 2);  // weight and bias
    EXPECT_TRUE(params[0].has_grad());
    EXPECT_TRUE(params[1].has_grad());
}

TEST_F(FlattenModuleTest, AutogradContiguousGradientCheck) {
    Tensor x = Tensor::randn({2, 3, 4});
    auto func = [](const std::vector<Tensor>& args) { return args[0].flatten(1, -1).sum(); };
    EXPECT_TRUE(gradient_check(func, {x}));
}

TEST_F(FlattenModuleTest, AutogradNonContiguousBackwardWeightedFormula) {
    Tensor base = Tensor::randn({2, 3, 4});
    base.set_requires_grad(true);

    Tensor transposed = base.transpose(0, 1);  // shape {3, 2, 4}
    Tensor flat = transposed.flatten(1, -1);   // shape {3, 8}

    // Weight matrix 3x8 with unique values: weights[j, c] = 8*j + c + 1
    std::vector<float> w_data(24);
    for (size_t j = 0; j < 3; ++j) {
        for (size_t c = 0; c < 8; ++c) {
            w_data[j * 8 + c] = static_cast<float>(j * 8 + c + 1);
        }
    }
    Tensor weights(w_data, Shape{3, 8});

    // Loss = sum(flat * weights)
    Tensor loss = (flat * weights).sum();
    loss.backward();

    ASSERT_TRUE(base.has_grad());
    EXPECT_EQ(base.grad().shape(), base.shape());

    // Analytical derivative check:
    // flat[j, i * 4 + k] corresponds to base[i, j, k]
    // dLoss / d(base[i, j, k]) = weights[j, i * 4 + k] = 8 * j + (4 * i + k) + 1
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                const float expected_grad = static_cast<float>(8 * j + 4 * i + k + 1);
                EXPECT_FLOAT_EQ(base.grad().item({i, j, k}), expected_grad);
            }
        }
    }
}
