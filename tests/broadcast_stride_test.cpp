#include <gtest/gtest.h>

#include "core/tensor.hpp"

using namespace helix;

TEST(BroadcastStrideTest, ZeroStride) {
    Tensor a({2, 1, 4});  // Stride will be (4, 4, 1)

    // Broadcast to (2, 3, 4)
    Tensor b = a.broadcast_to(Shape{2, 3, 4});

    EXPECT_EQ(b.shape(), Shape({2, 3, 4}));
    EXPECT_EQ(b.stride().vec(), std::vector<size_t>({4, 0, 1}));
}

TEST(BroadcastStrideTest, RankIncrease) {
    Tensor a({4});  // Stride (1)

    // Broadcast to (2, 3, 4)
    Tensor b = a.broadcast_to(Shape{2, 3, 4});

    EXPECT_EQ(b.shape(), Shape({2, 3, 4}));
    EXPECT_EQ(b.stride().vec(), std::vector<size_t>({0, 0, 1}));
}

TEST(BroadcastStrideTest, NonContiguousBroadcast) {
    // Create (3, 4), then transpose to (4, 3)
    Tensor a({3, 4});                // Stride (4, 1)
    Tensor a_t = a.transpose(0, 1);  // Shape (4, 3), Stride (1, 4)

    // Broadcast to (2, 4, 3)
    Tensor b = a_t.broadcast_to(Shape{2, 4, 3});

    EXPECT_EQ(b.shape(), Shape({2, 4, 3}));
    // Stride should be (0, 1, 4)
    EXPECT_EQ(b.stride().vec(), std::vector<size_t>({0, 1, 4}));
}
