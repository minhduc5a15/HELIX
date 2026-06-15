#include <gtest/gtest.h>

#include "core/tensor.hpp"

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
