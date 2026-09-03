#include <gtest/gtest.h>

#include "core/tensor.hpp"

using namespace helix;

TEST(StressExtremeShapes, ZeroSizeTensor) {
    Tensor a = Tensor::ones({0, 10, 10});
    Tensor b = Tensor::ones({0, 10, 10});
    Tensor c = a + b;
    EXPECT_EQ(c.numel(), 0);
    EXPECT_EQ(c.shape(), Shape({0, 10, 10}));

    Tensor d = Tensor::zeros({10, 0});
    Tensor e = Tensor::zeros({0, 5});
    Tensor f = d.matmul(e);
    EXPECT_EQ(f.shape(), Shape({10, 5}));
}

TEST(StressExtremeShapes, OneByOneTensor) {
    Tensor a = Tensor::ones({1, 1});
    Tensor b = Tensor::ones({1, 1});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.item(), 1.0f);
}

TEST(StressExtremeShapes, UnalignedMatrixSizes) {
    // 13x17x19 is highly unaligned for AVX2 (not multiple of 4, 8, 16)
    Tensor a = Tensor::ones({13, 17});
    Tensor b = Tensor::ones({17, 19});
    Tensor c = a.matmul(b);
    EXPECT_EQ(c.shape(), Shape({13, 19}));

    for (size_t i = 0; i < 13; ++i) {
        for (size_t j = 0; j < 19; ++j) {
            EXPECT_EQ(c.item({i, j}), 17.0f);
        }
    }
}

TEST(StressExtremeShapes, HighRankTensor) {
    // 8-D tensor
    Tensor a = Tensor::ones({2, 2, 2, 2, 2, 2, 2, 2});
    Tensor b = Tensor::ones({2, 2, 2, 2, 2, 2, 2, 2});
    Tensor c = a + b;

    EXPECT_EQ(c.numel(), 256);
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(c.data_ptr()[i], 2.0f);
    }
}
