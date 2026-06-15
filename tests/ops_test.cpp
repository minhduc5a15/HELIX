#include <gtest/gtest.h>

#include "core/tensor.hpp"

using namespace helix;

TEST(OpsTest, BasicAdd) {
    Tensor a({1.0f, 2.0f, 3.0f}, Shape{3});
    Tensor b({4.0f, 5.0f, 6.0f}, Shape{3});
    Tensor c = a + b;

    EXPECT_EQ(c.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(c.item({0}), 5.0f);
    EXPECT_FLOAT_EQ(c.item({1}), 7.0f);
    EXPECT_FLOAT_EQ(c.item({2}), 9.0f);
}

TEST(OpsTest, BroadcastAdd) {
    Tensor a({1, 2, 3, 4, 5, 6}, Shape{2, 3});
    Tensor b({10, 20, 30}, Shape{3});

    Tensor c = a + b;

    EXPECT_EQ(c.shape(), Shape({2, 3}));
    EXPECT_FLOAT_EQ(c.item({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.item({1, 2}), 36.0f);
}

TEST(OpsTest, TransposedAdd) {
    Tensor a({1, 2, 3, 4}, Shape{2, 2});
    Tensor b({10, 20, 30, 40}, Shape{2, 2});

    Tensor a_t = a.transpose(0, 1);

    Tensor c = a_t + b;

    EXPECT_EQ(c.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(c.item({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 23.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 32.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 44.0f);
}

TEST(OpsTest, UnaryNeg) {
    Tensor a({1.0f, -2.0f, 3.0f}, Shape{3});
    Tensor b = -a;

    EXPECT_FLOAT_EQ(b.item({0}), -1.0f);
    EXPECT_FLOAT_EQ(b.item({1}), 2.0f);
    EXPECT_FLOAT_EQ(b.item({2}), -3.0f);
}
