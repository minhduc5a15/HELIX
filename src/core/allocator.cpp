#include "core/allocator.hpp"

#include <cstdlib>
#include <new>
#include <ranges>

namespace helix {

    MemoryPool::~MemoryPool() { reset(); }

    void* MemoryPool::allocate(const size_t bytes) {
        if (bytes == 0) return nullptr;

        // 32-byte alignment for SIMD operations (AVX2/AVX-512)
        size_t alloc_size = (bytes + 31) & ~31;

        auto& list = free_blocks_[alloc_size];
        if (!list.empty()) {
            void* ptr = list.back();
            list.pop_back();
            return ptr;
        }

#if defined(_WIN32)
        void* ptr = _aligned_malloc(alloc_size, 32);
#else
        void* ptr = std::aligned_alloc(32, alloc_size);
#endif

        if (!ptr) throw std::bad_alloc();
        return ptr;
    }

    void MemoryPool::deallocate(void* ptr, const size_t bytes) {
        if (!ptr) return;
        const size_t alloc_size = (bytes + 31) & ~31;
        free_blocks_[alloc_size].push_back(ptr);
    }

    void MemoryPool::reset() {
        for (auto& list : free_blocks_ | std::views::values) {
            for (void* ptr : list) {
#if defined(_WIN32)
                _aligned_free(ptr);
#else
                std::free(ptr);
#endif
            }
        }
        free_blocks_.clear();
    }

    MemoryPool& MemoryPool::global() {
        static MemoryPool instance;
        return instance;
    }

}  // namespace helix
