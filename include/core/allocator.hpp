#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace helix {

    // Base interface for memory allocators
    class Allocator {
    public:
        virtual ~Allocator() = default;
        virtual void* allocate(size_t bytes) = 0;
        virtual void deallocate(void* ptr, size_t bytes) = 0;
    };

    struct GlobalBin {
        std::mutex mutex;
        std::vector<void*> blocks;
    };

    struct ThreadCache {
        std::unordered_map<size_t, std::vector<void*>> blocks;
    };

    // A caching Memory Pool that uses a TCMalloc-style architecture:
    // Thread-Local Cache for fast lock-free allocation, and a Global Pool
    // for garbage collection and cross-thread deallocation handling.
    class MemoryPool : public Allocator {
    public:
        MemoryPool();
        ~MemoryPool() override;

        void* allocate(size_t bytes) override;
        void deallocate(void* ptr, size_t bytes) override;

        // Frees all cached memory back to the OS
        void reset();

        // Global singleton accessor
        static MemoryPool& get_instance();

        // Diagnostic tool: returns the number of blocks currently in the global pool for a given size.
        size_t get_global_pool_size(size_t alloc_size);

    private:
        friend struct ThreadCacheWrapper;

        std::unordered_map<size_t, std::unique_ptr<GlobalBin>> global_bins_;
        std::mutex global_map_mutex_;

        std::unordered_set<ThreadCache*> all_caches_;
        std::mutex caches_mutex_;

        static constexpr size_t MAX_LOCAL_CACHE_SIZE = 64;
        static constexpr size_t TRANSFER_BATCH_SIZE = 32;

        GlobalBin* get_global_bin(size_t alloc_size);
        ThreadCache* get_thread_cache();
    };

}  // namespace helix
