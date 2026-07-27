#include <omp.h>

#include <iostream>

#include "core/allocator.hpp"

using namespace helix;

int main() {
    std::cout << "Starting MemoryPool thread-safety stress test with OpenMP..." << std::endl;

    const int NUM_THREADS = 16;  // Use many threads to increase chance of collision
    const int NUM_ITERATIONS = 100000;

    omp_set_num_threads(NUM_THREADS);

    // We will spam the MemoryPool from multiple threads simultaneously.
    // MemoryPool::get_instance().allocate() accesses the unordered_map `free_blocks_`
    // without any std::mutex. This will cause a data race.

#pragma omp parallel for
    for (int i = 0; i < NUM_THREADS * NUM_ITERATIONS; ++i) {
        // Pick sizes that vary to trigger map insertions (which rehash and invalidate iterators),
        // and also same sizes to trigger simultaneous push_back/pop_back on std::vector.
        size_t size = ((i % 1024) + 1) * 32;

        void* ptr = MemoryPool::get_instance().allocate(size);

        // Small delay to increase thread overlap
        for (volatile int j = 0; j < 50; ++j) {
        }

        MemoryPool::get_instance().deallocate(ptr, size);
    }

    std::cout << "If you see this, we got lucky and it didn't crash." << std::endl;
    return 0;
}
