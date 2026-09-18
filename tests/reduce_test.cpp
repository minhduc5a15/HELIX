#include <gtest/gtest.h>

#include "autograd/engine.hpp"
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

TEST(ReduceTest, ArgMax_Axis0) {
    // 2x3 Matrix:
    // [ [1, 5, 3],
    //   [4, 2, 6] ]
    std::vector<float> data = {1, 5, 3, 4, 2, 6};
    Tensor a(data, Shape{2, 3});

    Tensor idx = a.argmax(0);
    EXPECT_EQ(idx.shape().vec(), (std::vector<size_t>{3}));
    EXPECT_EQ(idx.dtype(), DType::Int64);

    const int64_t* ptr = idx.data_ptr<int64_t>();
    EXPECT_EQ(ptr[0], 1);  // 4 > 1
    EXPECT_EQ(ptr[1], 0);  // 5 > 2
    EXPECT_EQ(ptr[2], 1);  // 6 > 3
}

TEST(ReduceTest, ArgMax_Axis1) {
    // 2x3 Matrix:
    // [ [1, 5, 3],
    //   [4, 2, 6] ]
    std::vector<float> data = {1, 5, 3, 4, 2, 6};
    Tensor a(data, Shape{2, 3});

    Tensor idx = a.argmax(1);
    EXPECT_EQ(idx.shape().vec(), (std::vector<size_t>{2}));
    EXPECT_EQ(idx.dtype(), DType::Int64);

    const int64_t* ptr = idx.data_ptr<int64_t>();
    EXPECT_EQ(ptr[0], 1);  // max in row 0 is 5 at col 1
    EXPECT_EQ(ptr[1], 2);  // max in row 1 is 6 at col 2
}

TEST(ReduceTest, ArgMax_Rank1_ScalarOutput) {
    std::vector<float> data = {3.0f, 9.0f, 2.0f, 7.0f};
    Tensor a(data, Shape{4});

    Tensor idx = a.argmax(0);
    EXPECT_EQ(idx.rank(), 0);
    EXPECT_EQ(idx.numel(), 1);
    EXPECT_EQ(idx.dtype(), DType::Int64);

    const int64_t* ptr = idx.data_ptr<int64_t>();
    EXPECT_EQ(ptr[0], 1);
}

TEST(ReduceTest, ArgMax_TieBreaking) {
    // Multiple occurrences of maximum value -> must pick first index
    std::vector<float> data = {2.0f, 5.0f, 5.0f, 1.0f};
    Tensor a(data, Shape{4});

    Tensor idx = a.argmax(0);
    EXPECT_EQ(idx.data_ptr<int64_t>()[0], 1);
}

TEST(ReduceTest, ArgMax_NegativeValues) {
    std::vector<float> data = {-10.0f, -2.0f, -5.0f, -8.0f};
    Tensor a(data, Shape{4});

    Tensor idx = a.argmax(0);
    EXPECT_EQ(idx.data_ptr<int64_t>()[0], 1);
}

TEST(ReduceTest, ArgMax_AllDTypes) {
    // Float64
    Tensor d(Shape{3}, DType::Float64);
    d.data_ptr<double>()[0] = 1.0;
    d.data_ptr<double>()[1] = 5.0;
    d.data_ptr<double>()[2] = 2.0;
    EXPECT_EQ(d.argmax(0).data_ptr<int64_t>()[0], 1);

    // Int32
    Tensor i32(Shape{3}, DType::Int32);
    i32.data_ptr<int32_t>()[0] = 10;
    i32.data_ptr<int32_t>()[1] = 2;
    i32.data_ptr<int32_t>()[2] = 30;
    EXPECT_EQ(i32.argmax(0).data_ptr<int64_t>()[0], 2);

    // Int64
    Tensor i64(Shape{3}, DType::Int64);
    i64.data_ptr<int64_t>()[0] = 500;
    i64.data_ptr<int64_t>()[1] = 200;
    i64.data_ptr<int64_t>()[2] = 100;
    EXPECT_EQ(i64.argmax(0).data_ptr<int64_t>()[0], 0);
}

TEST(ReduceTest, ArgMax_NonContiguous) {
    // 2x3 transposed to 3x2:
    // a = [ [1, 5, 3],
    //       [4, 2, 6] ]
    // a.t() = [ [1, 4],
    //           [5, 2],
    //           [3, 6] ]
    std::vector<float> data = {1, 5, 3, 4, 2, 6};
    Tensor a(data, Shape{2, 3});
    Tensor a_t = a.transpose(0, 1);
    EXPECT_FALSE(a_t.is_contiguous());

    Tensor idx = a_t.argmax(1);
    EXPECT_EQ(idx.shape().vec(), (std::vector<size_t>{3}));
    const int64_t* ptr = idx.data_ptr<int64_t>();
    EXPECT_EQ(ptr[0], 1);  // max of [1, 4] is 4 at col 1
    EXPECT_EQ(ptr[1], 0);  // max of [5, 2] is 5 at col 0
    EXPECT_EQ(ptr[2], 1);  // max of [3, 6] is 6 at col 1
}

TEST(ReduceTest, ArgMax_ErrorValidation) {
    Tensor a({1.0f, 2.0f}, Shape{2});
    EXPECT_THROW(a.argmax(1), std::out_of_range);

    Tensor empty_dim(Shape{0, 2});
    EXPECT_THROW(empty_dim.argmax(0), std::invalid_argument);
}

TEST(ReduceTest, ArgMax_NoGradient) {
    init_autograd();
    Tensor a({1.0f, 3.0f, 2.0f}, Shape{3});
    a.set_requires_grad(true);

    Tensor idx = a.argmax(0);
    EXPECT_FALSE(idx.requires_grad());
    EXPECT_EQ(idx.impl()->autograd_meta(), nullptr);
}
