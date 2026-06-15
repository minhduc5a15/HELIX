#pragma once

#include <unordered_map>
#include <vector>

namespace helix {

    // Base interface for memory allocators
    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void* allocate(size_t bytes) = 0;
        virtual void deallocate(void* ptr, size_t bytes) = 0;
    };

    // A caching Memory Pool that caches freed blocks by size to avoid frequent malloc calls.
    // Provides O(1) allocation/deallocation for reused sizes.
    class MemoryPool : public Allocator {
    public:
        MemoryPool() = default;
        ~MemoryPool() override;

        void* allocate(size_t bytes) override;
        void deallocate(void* ptr, size_t bytes) override;

        // Frees all cached memory back to the OS
        void reset();

        // Global singleton accessor
        static MemoryPool& global();

    private:
        std::unordered_map<size_t, std::vector<void*>> free_blocks_;
    };

}  // namespace helix
