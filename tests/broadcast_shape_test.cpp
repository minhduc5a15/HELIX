#include <gtest/gtest.h>

#include "core/broadcast.hpp"

using namespace helix;

TEST(BroadcastShapeTest, SameShape) {
    Shape a{2, 3, 4};
    Shape b{2, 3, 4};
    Shape out = compute_broadcast_shape(a, b);
    EXPECT_EQ(out, Shape({2, 3, 4}));
}

TEST(BroadcastShapeTest, ScalarBroadcast) {
    Shape a{2, 3, 4};
    Shape b{};  // Scalar
    Shape out = compute_broadcast_shape(a, b);
    EXPECT_EQ(out, Shape({2, 3, 4}));
}

TEST(BroadcastShapeTest, AddDimensionRightToLeft) {
    Shape a{2, 3, 4};
    Shape b{4};  // Rank 1
    Shape out = compute_broadcast_shape(a, b);
    EXPECT_EQ(out, Shape({2, 3, 4}));
}

TEST(BroadcastShapeTest, ComplexBroadcast) {
    // (2, 1, 4) and (3, 4) -> (2, 3, 4)
    Shape a{2, 1, 4};
    Shape b{3, 4};
    Shape out = compute_broadcast_shape(a, b);
    EXPECT_EQ(out, Shape({2, 3, 4}));
}

TEST(BroadcastShapeTest, ZeroSizedDimensionBroadcastWithImplicitLeadingOne) {
    const Shape empty_batch{2, 0, 3};
    const Shape channel_values{3};

    EXPECT_EQ(compute_broadcast_shape(empty_batch, channel_values), Shape({2, 0, 3}));
    EXPECT_EQ(compute_broadcast_shape(channel_values, empty_batch), Shape({2, 0, 3}));
}

TEST(BroadcastShapeTest, ZeroSizedDimensionBroadcastWithExplicitOne) {
    const Shape empty_rows{4, 0, 7};
    const Shape singleton_rows{4, 1, 7};

    EXPECT_EQ(compute_broadcast_shape(empty_rows, singleton_rows), Shape({4, 0, 7}));
    EXPECT_EQ(compute_broadcast_shape(singleton_rows, empty_rows), Shape({4, 0, 7}));
}

TEST(BroadcastShapeTest, IncompatibleShapes) {
    Shape a{2, 3, 4};
    Shape b{2, 4};  // The last dim matches, but 3 != 2. Wait: right-to-left: 4==4, 3!=2 -> error
    EXPECT_THROW(compute_broadcast_shape(a, b), std::invalid_argument);
}
