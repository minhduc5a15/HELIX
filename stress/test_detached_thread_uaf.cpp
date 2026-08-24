#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "core/allocator.hpp"

using namespace helix;

// Test that detached threads allocating/deallocating during program exit do not cause UAF.
// NOTE: Since this involves detached threads outliving main(), the actual verification
// happens when the gtest framework exits the process. If a UAF occurs, the OS will segfault
// and the test run will crash.
TEST(AllocatorStressTest, DetachedThreadUAF) {
    auto start_flag = std::make_shared<std::atomic<bool>>(false);

    auto thread_func = [start_flag]() {
        // Wait for all threads to start simultaneously
        while (!start_flag->load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        // Vigorously allocate and deallocate
        for (int i = 0; i < 5000; ++i) {
            void* ptr1 = MemoryPool::get_instance().allocate(128);
            void* ptr2 = MemoryPool::get_instance().allocate(256);
            void* ptr3 = MemoryPool::get_instance().allocate(1024);

            // Add a tiny delay to ensure some threads survive past main()
            if (i % 1000 == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            MemoryPool::get_instance().deallocate(ptr1, 128);
            MemoryPool::get_instance().deallocate(ptr2, 256);
            MemoryPool::get_instance().deallocate(ptr3, 1024);
        }
    };

    // Spawn 50 detached threads
    for (int i = 0; i < 50; ++i) {
        std::thread t(thread_func);
        t.detach();  // Detach to let them outlive main
    }

    // Release the hounds
    start_flag->store(true, std::memory_order_release);

    // Give them a brief moment to get into the hot loop before main() returns
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // As main() exits, it will trigger exit handlers and destroy static variables.
    // The detached threads will still be running and accessing MemoryPool::get_instance().
    // If the UAF fix is correct, this will not segfault.
}
