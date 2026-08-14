#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#define private public
#include "core/allocator.hpp"
#undef private

using namespace helix;

// Define the variable here so it links, since it's missing in allocator.cpp
std::atomic<size_t> g_total_allocated{0};

TEST(AllocatorTest, TotalAllocatedIsNotUpdated) {
    size_t initial = g_total_allocated.load();
    MemoryPool& pool = MemoryPool::get_instance();

    void* ptr = pool.allocate(1024);
    EXPECT_EQ(g_total_allocated.load(), initial) << "BUG: g_total_allocated is not updated by allocate()!";

    pool.deallocate(ptr, 1024);
    EXPECT_EQ(g_total_allocated.load(), initial) << "BUG: g_total_allocated is not updated by deallocate()!";
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

    // Check if the memory blocks were pushed to global bin
    GlobalBin* bin = pool.get_global_bin(256);
    std::lock_guard<std::mutex> lock(bin->mutex);
    EXPECT_EQ(bin->blocks.size(), 2) << "Blocks should be in global bin after thread exit.";
}
