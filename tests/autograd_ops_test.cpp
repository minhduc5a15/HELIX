#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "grad_check.hpp"

using namespace helix;

class AutogradOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the Autograd Engine Hooks
        init_autograd();
    }
};

// 1. Broadcasting Backward Utility
TEST_F(AutogradOpsTest, AddBroadcastGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({5.0f, 6.0f}, Shape{1, 2});

    auto func = [](const std::vector<Tensor>& inputs) { return (inputs[0] + inputs[1]).sum(); };

    EXPECT_TRUE(gradient_check(func, {a, b}, 1e-3f, 5e-3f));
}

TEST_F(AutogradOpsTest, MulBroadcastGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({2.0f}, Shape{1});  // scalar tensor broadcasted

    auto func = [](const std::vector<Tensor>& inputs) { return (inputs[0] * inputs[1]).sum(); };

    EXPECT_TRUE(gradient_check(func, {a, b}, 1e-3f, 5e-3f));
}

// 2. Core Forward Operations (Unary Ops)
TEST_F(AutogradOpsTest, ExpGradientCheck) {
    Tensor a({0.1f, 0.5f, 1.0f, -0.5f}, Shape{2, 2});
    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].exp().sum(); };
    EXPECT_TRUE(gradient_check(func, {a}));
}

TEST_F(AutogradOpsTest, LogGradientCheck) {
    Tensor a({0.1f, 0.5f, 1.0f, 2.0f}, Shape{2, 2});  // inputs > 0 for log
    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].log().sum(); };
    EXPECT_TRUE(gradient_check(func, {a}));
}

TEST_F(AutogradOpsTest, SqrtGradientCheck) {
    Tensor a({0.1f, 0.5f, 1.0f, 2.0f}, Shape{2, 2});  // inputs > 0 for sqrt
    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].sqrt().sum(); };
    EXPECT_TRUE(gradient_check(func, {a}, 1e-4f, 5e-3f));
}

TEST_F(AutogradOpsTest, PowGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].pow(3.0f).sum(); };
    // Numerical precision can be tricky for larger powers, relax tolerance slightly or use smaller eps
    EXPECT_TRUE(gradient_check(func, {a}, 1e-3f, 5e-2f));
}

// 3. Reduction Backward (Sum, Mean with axis and keepdim)
TEST_F(AutogradOpsTest, SumAxisGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});
    auto func = [](const std::vector<Tensor>& inputs) {
        // sum over axis 1, then sum all to scalar
        return inputs[0].sum(1, false).sum();
    };
    EXPECT_TRUE(gradient_check(func, {a}));
}

TEST_F(AutogradOpsTest, MeanAxisKeepdimGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});
    auto func = [](const std::vector<Tensor>& inputs) {
        // mean over axis 0, keepdim=true, then sum all to scalar
        return inputs[0].mean(0, true).sum();
    };
    EXPECT_TRUE(gradient_check(func, {a}));
}

// 4. Additional Operations
TEST_F(AutogradOpsTest, SubBroadcastGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({5.0f, 6.0f}, Shape{1, 2});
    auto func = [](const std::vector<Tensor>& inputs) { return (inputs[0] - inputs[1]).sum(); };
    EXPECT_TRUE(gradient_check(func, {a, b}, 1e-4f, 5e-3f));
}

TEST_F(AutogradOpsTest, DivBroadcastGradientCheck) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({2.0f, 4.0f}, Shape{1, 2});
    auto func = [](const std::vector<Tensor>& inputs) { return (inputs[0] / inputs[1]).sum(); };
    EXPECT_TRUE(gradient_check(func, {a, b}, 1e-4f, 5e-3f));
}

TEST_F(AutogradOpsTest, MatMulGradientCheck) {
    Tensor a = Tensor::randn({3, 4});
    Tensor b = Tensor::randn({4, 2});
    auto func = [](const std::vector<Tensor>& inputs) { return inputs[0].matmul(inputs[1]).sum(); };
    EXPECT_TRUE(gradient_check(func, {a, b}, 1e-4f, 5e-3f));
}
