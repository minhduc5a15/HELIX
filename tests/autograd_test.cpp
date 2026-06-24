#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"

using namespace helix;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the Autograd Engine Hooks
        init_autograd();
    }
};

TEST_F(AutogradTest, SimpleAddMul) {
    Tensor a({2.0f}, Shape{1});
    Tensor b({3.0f}, Shape{1});

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    // c = a + b * a -> c = 2 + 3 * 2 = 8
    Tensor c = a + b * a;

    EXPECT_FLOAT_EQ(c.item(), 8.0f);

    c.backward();

    // dc/da = 1 + b = 4
    // dc/db = a = 2
    EXPECT_FLOAT_EQ(a.grad().item(), 4.0f);
    EXPECT_FLOAT_EQ(b.grad().item(), 2.0f);
}

TEST_F(AutogradTest, MeanGradient) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    a.set_requires_grad(true);

    Tensor c = a.mean();  // c = 2.5

    EXPECT_FLOAT_EQ(c.item(), 2.5f);

    c.backward();

    // dc/da = 1/4 = 0.25 for all elements
    const float* grad_ptr = a.grad().data_ptr();
    EXPECT_FLOAT_EQ(grad_ptr[0], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[1], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[2], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[3], 0.25f);
}

TEST_F(AutogradTest, Detach) {
    Tensor a({2.0f}, Shape{1});
    a.set_requires_grad(true);

    Tensor b = a.detach();
    EXPECT_FALSE(b.requires_grad());

    Tensor c = a * b;
    c.backward();

    // c = a * 2.0 (since b is detached, b is constant 2.0)
    // dc/da = b = 2.0
    EXPECT_FLOAT_EQ(a.grad().item(), 2.0f);
}
