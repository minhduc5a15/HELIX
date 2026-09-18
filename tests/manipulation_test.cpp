#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "core/tensor_factory.hpp"

using namespace helix;

TEST(ManipulationTest, Flatten) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});

    Tensor flat = a.flatten();

    EXPECT_EQ(flat.rank(), 1);
    EXPECT_EQ(flat.shape()[0], 6);
    EXPECT_TRUE(flat.is_contiguous());

    EXPECT_FLOAT_EQ(flat.item({0}), 1.0f);
    EXPECT_FLOAT_EQ(flat.item({5}), 6.0f);
}

TEST(ManipulationTest, SliceRow) {
    // 3x4 Matrix
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor a(data, Shape{3, 4});

    // Slice row 1 to 3 (exclusive), i.e., row 1 and 2
    Tensor sliced = a.slice(0, 1, 3);

    EXPECT_EQ(sliced.shape().vec(), (std::vector<size_t>{2, 4}));
    // Since we slice along the first dimension (row), the remaining rows are still contiguous
    EXPECT_TRUE(sliced.is_contiguous());

    EXPECT_FLOAT_EQ(sliced.item({0, 0}), 5.0f);
    EXPECT_FLOAT_EQ(sliced.item({1, 3}), 12.0f);
}

TEST(ManipulationTest, SliceCol) {
    // 3x4 Matrix
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor a(data, Shape{3, 4});

    // Slice col 1 to 3 (exclusive), i.e., col 1 and 2
    Tensor sliced = a.slice(1, 1, 3);

    EXPECT_EQ(sliced.shape().vec(), (std::vector<size_t>{3, 2}));
    // Slicing columns breaks contiguity
    EXPECT_FALSE(sliced.is_contiguous());

    EXPECT_FLOAT_EQ(sliced.item({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(sliced.item({2, 1}), 11.0f);

    // Ensure contiguous works on sliced tensor
    Tensor contig = sliced.contiguous();
    EXPECT_TRUE(contig.is_contiguous());
    EXPECT_FLOAT_EQ(contig.item({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(contig.item({2, 1}), 11.0f);
}

TEST(ManipulationTest, Cat_Dim0) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, Shape{3, 2});

    Tensor c = cat({a, b}, 0);
    EXPECT_EQ(c.shape().vec(), (std::vector<size_t>{5, 2}));
    EXPECT_TRUE(c.is_contiguous());

    EXPECT_FLOAT_EQ(c.item({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(c.item({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(c.item({2, 1}), 6.0f);
    EXPECT_FLOAT_EQ(c.item({4, 1}), 10.0f);
}

TEST(ManipulationTest, Cat_Dim1) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, Shape{2, 3});

    Tensor c = cat({a, b}, 1);
    EXPECT_EQ(c.shape().vec(), (std::vector<size_t>{2, 5}));
    EXPECT_TRUE(c.is_contiguous());

    EXPECT_FLOAT_EQ(c.item({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(c.item({0, 2}), 5.0f);
    EXPECT_FLOAT_EQ(c.item({0, 3}), 6.0f);
    EXPECT_FLOAT_EQ(c.item({0, 4}), 7.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(c.item({1, 2}), 8.0f);
    EXPECT_FLOAT_EQ(c.item({1, 3}), 9.0f);
    EXPECT_FLOAT_EQ(c.item({1, 4}), 10.0f);
}

TEST(ManipulationTest, Cat_ThreeTensors) {
    Tensor a({1.0f, 2.0f}, Shape{2});
    Tensor b({3.0f, 4.0f, 5.0f}, Shape{3});
    Tensor c({6.0f}, Shape{1});

    Tensor out = cat({a, b, c}, 0);
    EXPECT_EQ(out.shape().vec(), (std::vector<size_t>{6}));
    EXPECT_FLOAT_EQ(out.item({0}), 1.0f);
    EXPECT_FLOAT_EQ(out.item({1}), 2.0f);
    EXPECT_FLOAT_EQ(out.item({2}), 3.0f);
    EXPECT_FLOAT_EQ(out.item({3}), 4.0f);
    EXPECT_FLOAT_EQ(out.item({4}), 5.0f);
    EXPECT_FLOAT_EQ(out.item({5}), 6.0f);
}

TEST(ManipulationTest, Cat_AllDTypes) {
    // Float64
    Tensor d1(Shape{2}, DType::Float64);
    d1.data_ptr<double>()[0] = 1.5;
    d1.data_ptr<double>()[1] = 2.5;
    Tensor d2(Shape{1}, DType::Float64);
    d2.data_ptr<double>()[0] = 3.5;
    Tensor d_out = cat({d1, d2}, 0);
    EXPECT_EQ(d_out.dtype(), DType::Float64);
    EXPECT_EQ(d_out.shape()[0], 3);
    EXPECT_DOUBLE_EQ(d_out.data_ptr<double>()[0], 1.5);
    EXPECT_DOUBLE_EQ(d_out.data_ptr<double>()[2], 3.5);

    // Int32
    Tensor i1(Shape{2}, DType::Int32);
    i1.data_ptr<int32_t>()[0] = 10;
    i1.data_ptr<int32_t>()[1] = 20;
    Tensor i2(Shape{2}, DType::Int32);
    i2.data_ptr<int32_t>()[0] = 30;
    i2.data_ptr<int32_t>()[1] = 40;
    Tensor i_out = cat({i1, i2}, 0);
    EXPECT_EQ(i_out.dtype(), DType::Int32);
    EXPECT_EQ(i_out.data_ptr<int32_t>()[0], 10);
    EXPECT_EQ(i_out.data_ptr<int32_t>()[3], 40);

    // Int64
    Tensor l1(Shape{1}, DType::Int64);
    l1.data_ptr<int64_t>()[0] = 100;
    Tensor l2(Shape{1}, DType::Int64);
    l2.data_ptr<int64_t>()[0] = 200;
    Tensor l_out = cat({l1, l2}, 0);
    EXPECT_EQ(l_out.dtype(), DType::Int64);
    EXPECT_EQ(l_out.data_ptr<int64_t>()[0], 100);
    EXPECT_EQ(l_out.data_ptr<int64_t>()[1], 200);
}

TEST(ManipulationTest, Cat_NonContiguousInput) {
    // 3x4 Matrix
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor orig(data, Shape{3, 4});

    // Sliced column 1 to 3 -> shape {3, 2}, non-contiguous
    Tensor non_contig = orig.slice(1, 1, 3);
    EXPECT_FALSE(non_contig.is_contiguous());

    Tensor contig = Tensor::ones(Shape{3, 1});

    // Cat along dim 1: {3, 2} + {3, 1} -> {3, 3}
    Tensor res = cat({non_contig, contig}, 1);
    EXPECT_EQ(res.shape().vec(), (std::vector<size_t>{3, 3}));
    EXPECT_TRUE(res.is_contiguous());

    // Row 0: [2, 3, 1]
    EXPECT_FLOAT_EQ(res.item({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(res.item({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(res.item({0, 2}), 1.0f);

    // Row 2: [10, 11, 1]
    EXPECT_FLOAT_EQ(res.item({2, 0}), 10.0f);
    EXPECT_FLOAT_EQ(res.item({2, 1}), 11.0f);
    EXPECT_FLOAT_EQ(res.item({2, 2}), 1.0f);
}

TEST(ManipulationTest, Cat_ValidationErrors) {
    // Empty list
    EXPECT_THROW(cat({}, 0), std::invalid_argument);

    // Dim out of range
    Tensor a({1.0f, 2.0f}, Shape{2});
    EXPECT_THROW(cat({a}, 1), std::out_of_range);

    // Rank mismatch
    Tensor b({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    EXPECT_THROW(cat({a, b}, 0), std::invalid_argument);

    // DType mismatch
    Tensor c(Shape{2}, DType::Int32);
    EXPECT_THROW(cat({a, c}, 0), std::invalid_argument);

    // Non-cat dimension mismatch
    Tensor d({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3});
    // b is {2, 2}, d is {2, 3}; cat along dim 0 requires dim 1 to match!
    EXPECT_THROW(cat({b, d}, 0), std::invalid_argument);
}
