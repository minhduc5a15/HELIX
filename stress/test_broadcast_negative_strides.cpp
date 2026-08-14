#include <gtest/gtest.h>
#include "core/tensor.hpp"

using namespace helix;

TEST(BroadcastTest, ScalarTailBug) {
    // avx2_matmul_block has a scalar tail loop that might be skipped or overwrites
    // Let's create a matrix with M=5, N=5, K=5 to trigger micro_kernel_4x16 (which won't fit)
    // then 4x8 (won't fit), then N scalar tail, and M scalar tail.
    Tensor a = Tensor::ones(Shape({5, 5}));
    Tensor b = Tensor::ones(Shape({5, 5}));
    
    Tensor c = a.matmul(b);
    
    // All elements should be 5.0
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_EQ(c.item({i, j}), 5.0f);
        }
    }
}

TEST(BroadcastTest, ZeroSizeTensor) {
    Tensor a = Tensor::empty(Shape({0, 5}));
    Tensor b = Tensor::ones(Shape({5, 5}));
    
    // matmul should throw or return zero-size tensor
    // Currently, it might crash or return invalid tensor
    try {
        Tensor c = a.matmul(b);
        EXPECT_EQ(c.shape().numel(), 0);
    } catch (...) {
        // Exception is also fine if it's explicitly handled
    }
}
