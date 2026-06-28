#include "optim/sgd.hpp"

#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "nn/linear.hpp"

using namespace helix;

class SGDTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(SGDTest, SharedStorageTest) {
    // Verify that optimizer updates parameters in place without copying.
    Linear linear(2, 2);

    // Set weight to identity, bias to zeros
    Tensor w = Tensor::zeros({2, 2});
    w.data_ptr()[0] = 1.0f;
    w.data_ptr()[3] = 1.0f;
    Tensor b = Tensor::zeros({2});

    // In-place copy into parameters
    linear.parameters()[0].copy_(w);
    linear.parameters()[1].copy_(b);

    // Retrieve references
    auto weight_ref = linear.named_parameters()[0].second;

    // Create optimizer
    SGD opt(linear.parameters(), 0.1f);

    // Mock gradient
    Tensor x = Tensor::ones({1, 2});
    Tensor y = linear(x);
    Tensor loss = y.sum();
    loss.backward();

    // Check initial weight value (1.0 at index 0)
    EXPECT_FLOAT_EQ(weight_ref.data_ptr()[0], 1.0f);

    // Perform optimizer step
    opt.step();

    // Check that linear weight has changed in place
    // input is [1, 1], grad of w is x^T * dy = [[1, 1]]^T * [[1, 1]] = [[1, 1], [1, 1]]
    // w_new = w - 0.1 * grad = 1.0 - 0.1 * 1.0 = 0.9
    EXPECT_FLOAT_EQ(weight_ref.data_ptr()[0], 0.9f);
}

TEST_F(SGDTest, ConstraintsValidationCheck) {
    // Tests exception throwing on mismatch
    Tensor param = Tensor::randn({2, 2});
    param.set_requires_grad(true);

    // 1. Throw if no grad exists yet
    SGD opt({param}, 0.1f);
    EXPECT_THROW({ opt.step(); }, std::runtime_error);

    // Initialize a valid grad first
    Tensor loss = param.sum();
    loss.backward();

    // 2. Shape mismatch (manually reassign grad Tensor to different shape)
    param.grad() = Tensor::randn({3, 3});
    EXPECT_THROW({ opt.step(); }, std::invalid_argument);
}

TEST_F(SGDTest, ZeroGradCheck) {
    Tensor param = Tensor::randn({2, 2});
    param.set_requires_grad(true);

    SGD opt({param}, 0.1f);

    // Initialize gradient
    Tensor x = param.sum();
    x.backward();

    // Gradient should be all ones
    EXPECT_FLOAT_EQ(param.grad().data_ptr()[0], 1.0f);

    opt.zero_grad();

    // Gradient should be all zeros
    EXPECT_FLOAT_EQ(param.grad().data_ptr()[0], 0.0f);
    EXPECT_FLOAT_EQ(param.grad().data_ptr()[1], 0.0f);
}

TEST_F(SGDTest, NonContiguousParameterUpdate) {
    // Param created non-contiguous
    Tensor base_param = Tensor::randn({2, 2});

    Tensor param_t = base_param.transpose(0, 1);
    EXPECT_FALSE(param_t.is_contiguous());

    // Manually set requires_grad to true for param_t
    param_t.set_requires_grad(true);

    SGD opt({param_t}, 0.1f);

    // Mock gradient (contiguous)
    Tensor loss = param_t.sum();
    loss.backward();

    // Perform step
    float val_before = param_t.data_ptr()[0];
    opt.step();
    float val_after = param_t.data_ptr()[0];

    // Since gradient of sum is 1.0, value should decrease by 0.1
    EXPECT_NEAR(val_after, val_before - 0.1f, 1e-5f);
}
