#include <gtest/gtest.h>

#include <functional>

#include "core/value.hpp"

using namespace helix;

// Helper to compute numerical gradient using finite differences
float compute_numerical_gradient(float& param, float eps, const std::function<Value()>& forward_pass) {
    float original = param;

    // f(x + eps)
    param = original + eps;
    Value out_plus = forward_pass();
    float y_plus = out_plus.data();

    // f(x - eps)
    param = original - eps;
    Value out_minus = forward_pass();
    float y_minus = out_minus.data();

    // Restore
    param = original;

    return (y_plus - y_minus) / (2.0f * eps);
}

TEST(GradientCheckTest, PolynomialExpression) {
    float x_val = 2.0f;
    float y_val = -3.0f;

    std::function<Value()> forward = [&]() -> Value {
        Value x(x_val);
        Value y(y_val);
        return x.pow(3.0f) * y + y.pow(2.0f) - x * Value(5.0f);
    };

    // Analytical Gradient (Autograd)
    Value x(x_val);
    Value y(y_val);
    Value out = x.pow(3.0f) * y + y.pow(2.0f) - x * Value(5.0f);
    out.backward();

    // Numerical Gradient
    float num_grad_x = compute_numerical_gradient(x_val, 1e-3f, forward);
    float num_grad_y = compute_numerical_gradient(y_val, 1e-3f, forward);

    // Check if difference is less than 1e-2
    EXPECT_NEAR(x.grad(), num_grad_x, 1e-2f);
    EXPECT_NEAR(y.grad(), num_grad_y, 1e-2f);
}
