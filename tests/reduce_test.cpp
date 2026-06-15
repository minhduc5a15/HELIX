#include <gtest/gtest.h>

#include "core/tensor.hpp"

using namespace helix;

TEST(ReduceTest, SumAll) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});

    Tensor s = a.sum();
    EXPECT_EQ(s.rank(), 0);
    EXPECT_FLOAT_EQ(s.item(), 10.0f);
}

TEST(ReduceTest, SumAxis0) {
    // 2x3 Matrix
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor a(data, Shape{2, 3});

    Tensor s = a.sum(0);
    EXPECT_EQ(s.shape().vec(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(s.item({0}), 5.0f);
    EXPECT_FLOAT_EQ(s.item({1}), 7.0f);
    EXPECT_FLOAT_EQ(s.item({2}), 9.0f);
}

TEST(ReduceTest, SumAxis1) {
    // 2x3 Matrix
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor a(data, Shape{2, 3});

    Tensor s = a.sum(1);
    EXPECT_EQ(s.shape().vec(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(s.item({0}), 6.0f);
    EXPECT_FLOAT_EQ(s.item({1}), 15.0f);
}

TEST(ReduceTest, MeanAxis0) {
    std::vector<float> data = {1, 2, 3, 5, 6, 7};
    Tensor a(data, Shape{2, 3});

    Tensor m = a.mean(0);
    EXPECT_EQ(m.shape().vec(), (std::vector<size_t>{3}));
    EXPECT_FLOAT_EQ(m.item({0}), 3.0f);
    EXPECT_FLOAT_EQ(m.item({1}), 4.0f);
    EXPECT_FLOAT_EQ(m.item({2}), 5.0f);
}

TEST(ReduceTest, KeepDim) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});

    Tensor s = a.sum(0, true);
    EXPECT_EQ(s.shape().vec(), (std::vector<size_t>{1, 2}));
    EXPECT_FLOAT_EQ(s.item({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(s.item({0, 1}), 6.0f);
}
