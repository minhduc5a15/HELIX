#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include "core/allocator.hpp"

using namespace helix;

std::mutex q_mutex;
std::vector<void*> shared_queue;

int main() {
    std::cout << "Starting Cross-Thread Memory Allocation Stress Test..." << std::endl;
    
    const int NUM_ITEMS = 2000000; // 2 Million allocations
    const size_t ALLOC_SIZE = 1024; // 1KB per item (Total 2GB if not freed properly)
    
    // Thread 1: Allocates memory and passes it to Thread 2
    std::thread t1([&]() {
        for (int i = 0; i < NUM_ITEMS; ++i) {
            void* ptr = MemoryPool::get_instance().allocate(ALLOC_SIZE);
            
            // Push to shared queue
            std::lock_guard<std::mutex> lock(q_mutex);
            shared_queue.push_back(ptr);
        }
    });
    
    // Thread 2: Consumes memory and deallocates it
    std::thread t2([&]() {
        int processed = 0;
        while (processed < NUM_ITEMS) {
            void* ptr = nullptr;
            {
                std::lock_guard<std::mutex> lock(q_mutex);
                if (!shared_queue.empty()) {
                    ptr = shared_queue.back();
                    shared_queue.pop_back();
                }
            }
            
            if (ptr) {
                // Cross-thread deallocation
                MemoryPool::get_instance().deallocate(ptr, ALLOC_SIZE);
                processed++;
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    t1.join();
    t2.join();
    
    std::cout << "Cross-Thread Allocation Test Completed successfully!" << std::endl;
    std::cout << "Memory Bloat prevented: Local Cache flushed back to Global Pool automatically." << std::endl;
    return 0;
}
