#include <gtest/gtest.h>

#include <atomic>

#include "core/tensor.hpp"
#include "core/tensor_iterator.hpp"

using namespace helix;

TEST(OpsTest, TensorIteratorCoalescesContiguousDimensions) {
    Tensor output(Shape{2, 3, 4});
    Tensor input(Shape{2, 3, 4});
    TensorIterator iterator(output, {&input});

    EXPECT_EQ(iterator.rank(), 1);
    EXPECT_EQ(iterator.numel(), 24);
    EXPECT_TRUE(iterator.is_contiguous());
}

TEST(OpsTest, TensorIteratorParallelBroadcastTraversal) {
    Tensor output(Shape{2, 3});
    Tensor input({10.0f, 20.0f, 30.0f}, Shape{3});
    TensorIterator iterator(output, {&input});
    std::atomic<size_t> visits = 0;

    iterator.for_each_parallel(
        [&](const size_t flat_index, const auto& offsets) {
            output.data_ptr()[flat_index] = input.data_ptr()[offsets[1]];
            visits.fetch_add(1, std::memory_order_relaxed);
        },
        1
    );

    EXPECT_EQ(visits, 6);
    EXPECT_FLOAT_EQ(output.item({0, 0}), 10.0f);
    EXPECT_FLOAT_EQ(output.item({0, 2}), 30.0f);
    EXPECT_FLOAT_EQ(output.item({1, 0}), 10.0f);
    EXPECT_FLOAT_EQ(output.item({1, 2}), 30.0f);
}

TEST(OpsTest, TensorIteratorRejectsZeroGrainSize) {
    Tensor output(Shape{2});
    TensorIterator iterator(output, {});

    EXPECT_THROW(iterator.for_each_parallel([](const size_t, const auto&) {}, 0), std::invalid_argument);
}

TEST(OpsTest, BasicAdd) {
    Tensor a({1.0f, 2.0f, 3.0f}, Shape{3});
    Tensor b({4.0f, 5.0f, 6.0f}, Shape{3});
    Tensor c = a + b;

    EXPECT_EQ(c.shape(), Shape({3}));
    EXPECT_FLOAT_EQ(c.item({0}), 5.0f);
    EXPECT_FLOAT_EQ(c.item({1}), 7.0f);
    EXPECT_FLOAT_EQ(c.item({2}), 9.0f);
}

TEST(OpsTest, BroadcastAdd) {
    Tensor a({1, 2, 3, 4, 5, 6}, Shape{2, 3});
    Tensor b({10, 20, 30}, Shape{3});

    Tensor c = a + b;

    EXPECT_EQ(c.shape(), Shape({2, 3}));
    EXPECT_FLOAT_EQ(c.item({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.item({1, 2}), 36.0f);
}

TEST(OpsTest, TransposedAdd) {
    Tensor a({1, 2, 3, 4}, Shape{2, 2});
    Tensor b({10, 20, 30, 40}, Shape{2, 2});

    Tensor a_t = a.transpose(0, 1);

    Tensor c = a_t + b;

    EXPECT_EQ(c.shape(), Shape({2, 2}));
    EXPECT_FLOAT_EQ(c.item({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.item({0, 1}), 23.0f);
    EXPECT_FLOAT_EQ(c.item({1, 0}), 32.0f);
    EXPECT_FLOAT_EQ(c.item({1, 1}), 44.0f);
}

TEST(OpsTest, BroadcastBinaryOpsUseStridedIterator) {
    Tensor a({2, 4, 6, 8}, Shape{2, 2});
    Tensor b({1, 2}, Shape{2});

    Tensor transposed = a.transpose(0, 1);
    Tensor subtracted = transposed - b;
    Tensor multiplied = transposed * b;
    Tensor divided = transposed / b;

    EXPECT_FLOAT_EQ(subtracted.item({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(subtracted.item({1, 1}), 6.0f);
    EXPECT_FLOAT_EQ(multiplied.item({0, 1}), 12.0f);
    EXPECT_FLOAT_EQ(multiplied.item({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(divided.item({0, 1}), 3.0f);
    EXPECT_FLOAT_EQ(divided.item({1, 1}), 4.0f);
}

TEST(OpsTest, UnaryNeg) {
    Tensor a({1.0f, -2.0f, 3.0f}, Shape{3});
    Tensor b = -a;

    EXPECT_FLOAT_EQ(b.item({0}), -1.0f);
    EXPECT_FLOAT_EQ(b.item({1}), 2.0f);
    EXPECT_FLOAT_EQ(b.item({2}), -3.0f);
}

TEST(OpsTest, TypePromotion_Int32_Float32_IsFloat32) {
    Tensor a(Shape({2}), DType::Int32);
    int32_t* ptr_a = a.data_ptr<int32_t>();
    ptr_a[0] = 5;
    ptr_a[1] = 10;

    Tensor b({2.5f, 3.5f}, Shape({2}));  // Float32 by default

    Tensor c = a + b;

    EXPECT_EQ(c.dtype(), DType::Float32);
    EXPECT_FLOAT_EQ(c.item({0}), 7.5f);
    EXPECT_FLOAT_EQ(c.item({1}), 13.5f);
}

TEST(OpsTest, TypePromotion_Int32_Int64_IsInt64) {
    Tensor a(Shape({2}), DType::Int32);
    int32_t* ptr_a = a.data_ptr<int32_t>();
    ptr_a[0] = -5;
    ptr_a[1] = 100;

    Tensor b(Shape({2}), DType::Int64);
    int64_t* ptr_b = b.data_ptr<int64_t>();
    ptr_b[0] = 15;
    ptr_b[1] = 200;

    Tensor c = a + b;

    EXPECT_EQ(c.dtype(), DType::Int64);
    int64_t* ptr_c = c.data_ptr<int64_t>();
    EXPECT_EQ(ptr_c[0], 10);
    EXPECT_EQ(ptr_c[1], 300);
}
