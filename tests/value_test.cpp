#include "core/value.hpp"

#include <gtest/gtest.h>

using namespace helix;

TEST(ValueTest, BasicAddition) {
    Value a(2.0f);
    Value b(3.0f);
    Value c = a + b;

    EXPECT_FLOAT_EQ(c.data(), 5.0f);
    c.backward();

    EXPECT_FLOAT_EQ(a.grad(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad(), 1.0f);
}

TEST(ValueTest, BasicMultiplication) {
    Value a(2.0f);
    Value b(3.0f);
    Value c = a * b;

    EXPECT_FLOAT_EQ(c.data(), 6.0f);
    c.backward();

    EXPECT_FLOAT_EQ(a.grad(), 3.0f);
    EXPECT_FLOAT_EQ(b.grad(), 2.0f);
}

TEST(ValueTest, GradientAccumulation) {
    Value a(3.0f);
    Value b = a + a;  // b = 2a
    b.backward();

    // grad of a should be 2.0
    EXPECT_FLOAT_EQ(a.grad(), 2.0f);
}

TEST(ValueTest, ComplexExpression) {
    Value a(-4.0f);
    Value b(2.0f);

    // c = a + b = -2.0
    Value c = a + b;

    // d = a * b + b^3 = -8.0 + 8.0 = 0.0
    Value d = a * b + b.pow(3.0f);

    // e = c * d = -2.0 * 0.0 = 0.0
    Value e = c * d;

    e.backward();
    EXPECT_FLOAT_EQ(e.data(), 0.0f);

    // dz/da
    // e = (a+b) * (a*b + b^3)
    // de/da = 1 * (a*b + b^3) + (a+b) * b = 0 + (-2) * 2 = -4
    EXPECT_FLOAT_EQ(a.grad(), -4.0f);
}
