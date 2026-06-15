#include <gtest/gtest.h>

#include "core/allocator.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/shape.hpp"
#include "core/storage.hpp"
#include "core/stride.hpp"

using namespace helix;

TEST(CoreTest, ShapeAndStride) {
    Shape s({3, 4, 5});
    EXPECT_EQ(s.rank(), 3);
    EXPECT_EQ(s.numel(), 60);
    EXPECT_EQ(s[0], 3);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(s[2], 5);

    Stride st = Stride::compute_contiguous(s);
    EXPECT_EQ(st.rank(), 3);
    EXPECT_EQ(st[0], 20);  // 4 * 5
    EXPECT_EQ(st[1], 5);   // 5
    EXPECT_EQ(st[2], 1);   // 1

    EXPECT_EQ(st.compute_offset({1, 2, 3}), 1 * 20 + 2 * 5 + 3 * 1);  // 20 + 10 + 3 = 33
}

TEST(CoreTest, ShapeScalar) {
    Shape s;  // Rank 0
    EXPECT_EQ(s.rank(), 0);
    EXPECT_EQ(s.numel(), 1);
    EXPECT_TRUE(s.empty());

    Stride st = Stride::compute_contiguous(s);
    EXPECT_EQ(st.rank(), 0);
    EXPECT_TRUE(st.empty());
}

TEST(CoreTest, MemoryPool) {
    auto& pool = MemoryPool::global();

    void* ptr1 = pool.allocate(100);
    EXPECT_NE(ptr1, nullptr);

    // Test 0-byte allocation
    void* ptr0 = pool.allocate(0);
    EXPECT_EQ(ptr0, nullptr);

    pool.deallocate(ptr1, 100);

    void* ptr2 = pool.allocate(100);
    EXPECT_EQ(ptr1, ptr2);  // Should reuse the same block

    pool.deallocate(ptr2, 100);
}

TEST(CoreTest, Storage) {
    Storage s(128);  // 128 bytes
    EXPECT_EQ(s.size_bytes(), 128);
    EXPECT_NE(s.data(), nullptr);

    // Test move semantics
    Storage s2(std::move(s));
    EXPECT_EQ(s.size_bytes(), 0);
    EXPECT_EQ(s.data(), nullptr);

    EXPECT_EQ(s2.size_bytes(), 128);
    EXPECT_NE(s2.data(), nullptr);
}
