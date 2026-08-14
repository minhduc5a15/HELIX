#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "core/allocator.hpp"

using namespace helix;

// Declare the variable to link against the core library
extern std::atomic<size_t> g_total_allocated;

TEST(AllocatorTest, TotalAllocatedIsTrackedCorrectly) {
    size_t initial = g_total_allocated.load();
    MemoryPool& pool = MemoryPool::get_instance();

    void* ptr = pool.allocate(1024);
    EXPECT_EQ(g_total_allocated.load(), initial + 1024) << "g_total_allocated should increase by 1024!";

    pool.deallocate(ptr, 1024);
    EXPECT_EQ(g_total_allocated.load(), initial) << "g_total_allocated should return to initial!";
}

TEST(AllocatorTest, ThreadExitLeak) {
    MemoryPool& pool = MemoryPool::get_instance();

    auto thread_func = [&]() {
        void* ptr1 = pool.allocate(256);
        void* ptr2 = pool.allocate(256);
        pool.deallocate(ptr1, 256);
        pool.deallocate(ptr2, 256);
    };

    std::thread t1(thread_func);
    t1.join();

    // Check if the memory blocks were pushed to global pool
    EXPECT_EQ(pool.get_global_pool_size(256), 2) << "Blocks should be in global pool after thread exit.";
}
