#include "nn/relu.hpp"

#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "grad_check.hpp"

using namespace helix;

class ReLUTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(ReLUTest, Forward) {
    Tensor x({4});
    x.data_ptr()[0] = -2.0f;
    x.data_ptr()[1] = -1.0f;
    x.data_ptr()[2] = 0.0f;
    x.data_ptr()[3] = 3.0f;

    Tensor y = relu(x);

    EXPECT_FLOAT_EQ(y.data_ptr()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[1], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[2], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[3], 3.0f);
}

TEST_F(ReLUTest, Backward) {
    Tensor x({2});
    x.data_ptr()[0] = -1.0f;
    x.data_ptr()[1] = 2.0f;
    x.set_requires_grad(true);

    Tensor y = relu(x);
    Tensor loss = y.sum();
    loss.backward();

    Tensor grad = x.grad();
    EXPECT_FLOAT_EQ(grad.data_ptr()[0], 0.0f);
    EXPECT_FLOAT_EQ(grad.data_ptr()[1], 1.0f);
}

TEST_F(ReLUTest, GradientCheck) {
    Tensor x = Tensor::randn({4, 4});
    // Add small offset to avoid exactly 0 (non-differentiable point)
    x = x + 0.1f;

    auto func = [](const std::vector<Tensor>& args) { return relu(args[0]).sum(); };

    EXPECT_TRUE(gradient_check(func, {x}));
}

TEST_F(ReLUTest, EdgeCases_AllNegativeOrZero) {
    Tensor x({4});
    x.data_ptr()[0] = -5.0f;
    x.data_ptr()[1] = -1.0f;
    x.data_ptr()[2] = 0.0f;
    x.data_ptr()[3] = -100.0f;
    x.set_requires_grad(true);

    Tensor y = relu(x);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(y.data_ptr()[i], 0.0f);
    }

    Tensor loss = y.sum();
    loss.backward();

    Tensor grad = x.grad();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(grad.data_ptr()[i], 0.0f);
    }
}

TEST_F(ReLUTest, Contiguity_NonContiguousInput) {
    Tensor x = Tensor::randn({4, 4});
    Tensor x_t = x.transpose(0, 1);
    x_t.set_requires_grad(true);

    Tensor y = relu(x_t);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{4, 4}));

    Tensor loss = y.sum();
    loss.backward();

    EXPECT_EQ(x_t.grad().shape().vec(), (std::vector<size_t>{4, 4}));
}

TEST_F(ReLUTest, ShallowCopyDynamics_NoSourceModification) {
    Tensor x({4});
    x.data_ptr()[0] = -1.0f;
    x.data_ptr()[1] = 2.0f;
    x.data_ptr()[2] = -3.0f;
    x.data_ptr()[3] = 4.0f;

    Tensor y = relu(x);

    // Ensure source is not modified
    EXPECT_FLOAT_EQ(x.data_ptr()[0], -1.0f);
    EXPECT_FLOAT_EQ(x.data_ptr()[1], 2.0f);
    EXPECT_FLOAT_EQ(x.data_ptr()[2], -3.0f);
    EXPECT_FLOAT_EQ(x.data_ptr()[3], 4.0f);
}

TEST_F(ReLUTest, ReLU) {
    // Explicitly verify ReLU forward activation values
    ReLU relu_layer;
    
    Tensor x({5});
    x.data_ptr()[0] = -10.0f;
    x.data_ptr()[1] = 0.0f;
    x.data_ptr()[2] = 5.0f;
    x.data_ptr()[3] = -0.5f;
    x.data_ptr()[4] = 100.0f;
    
    Tensor y = relu_layer(x);
    
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{5}));
    EXPECT_FLOAT_EQ(y.data_ptr()[0], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[1], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[2], 5.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[3], 0.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[4], 100.0f);
}

