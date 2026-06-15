#include <gtest/gtest.h>

#include "core/tensor.hpp"

using namespace helix;

TEST(MatmulTest, Basic2D) {
    // 2x3 matrix
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});

    // 3x2 matrix
    Tensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, Shape{3, 2});

    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape(), Shape({2, 2}));
    // Row 0, Col 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // Row 0, Col 1: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // Row 1, Col 0: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // Row 1, Col 1: 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    EXPECT_FLOAT_EQ(c.item({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 154.0f);
}

TEST(MatmulTest, TransposedInputs) {
    // b is initially 2x3, but we want to multiply a (2x3) with b^T (3x2)
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});

    Tensor b({7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f}, Shape{2, 3});

    Tensor b_t = b.transpose(0, 1);  // shape is now 3x2, identical mathematically to 'b' from previous test

    Tensor c = a.matmul(b_t);

    EXPECT_EQ(c.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(c.item({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 154.0f);
}

TEST(MatmulTest, IncompatibleShapes) {
    Tensor a({1, 2}, Shape{2});  // Rank 1
    Tensor b({1, 2}, Shape{2});
    EXPECT_THROW(a.matmul(b), std::invalid_argument);

    Tensor c({1, 2}, Shape{1, 2});
    Tensor d({1, 2, 3}, Shape{1, 3});
    EXPECT_THROW(c.matmul(d), std::invalid_argument);  // 2 != 1
}
