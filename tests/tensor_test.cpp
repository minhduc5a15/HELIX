#include "core/tensor.hpp"

#include <gtest/gtest.h>

using namespace helix;

TEST(TensorTest, DefaultConstructor) {
    Tensor t;
    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_EQ(t.shape().empty(), true);
}

TEST(TensorTest, ConstructorWithShape) {
    Shape s({2, 3});
    Tensor t(s);
    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(TensorTest, ConstructorWithData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Shape s({2, 3});
    Tensor t(data, s);

    EXPECT_EQ(t.numel(), 6);
    EXPECT_FLOAT_EQ(t.item({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.item({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.item({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(t.item({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(t.item({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(t.item({1, 2}), 6.0f);
}

TEST(TensorTest, SetItem) {
    Shape s({2});
    Tensor t(s);

    t.set_item({0}, 42.0f);
    t.set_item({1}, 100.0f);

    EXPECT_FLOAT_EQ(t.item({0}), 42.0f);
    EXPECT_FLOAT_EQ(t.item({1}), 100.0f);
}

TEST(TensorTest, ShallowCopy) {
    std::vector<float> data = {1.0f, 2.0f};
    Shape s({2});
    Tensor t1(data, s);
    Tensor t2 = t1;  // Shallow copy

    EXPECT_EQ(t1.data_ptr(), t2.data_ptr());

    t2.set_item({0}, 99.0f);
    EXPECT_FLOAT_EQ(t1.item({0}), 99.0f);
}
