#include <atomic>
// An atomic variable to track the total memory currently allocated by the custom allocator.
// This is useful for debugging and monitoring memory usage across the application.
std::atomic<size_t> g_total_allocated{0};

#include <algorithm>  // For std::min
#include <cstdlib>    // For std::aligned_alloc and std::free
#include <new>        // For std::bad_alloc

#include "core/allocator.hpp"  // Custom allocator declarations

namespace helix {

    /**
     * @brief Constructor for MemoryPool.
     * Initializes a new memory pool instance. In this case, it uses the default constructor.
     */
    MemoryPool::MemoryPool() = default;

    /**
     * @brief Destructor for MemoryPool.
     * Cleans up all allocated memory when the MemoryPool instance is destroyed.
     * This is crucial to prevent memory leaks.
     */
    MemoryPool::~MemoryPool() { reset(); }

    /**
     * @brief Retrieves or creates a GlobalBin for a specific allocation size.
     *
     * GlobalBins manage memory blocks of a particular size that are shared across all threads.
     * This function ensures that only one GlobalBin exists per allocation size.
     *
     * @param alloc_size The standardized size of the memory blocks to manage.
     * @return A pointer to the GlobalBin for the given allocation size.
     */
    GlobalBin* MemoryPool::get_global_bin(size_t alloc_size) {
        // Protects access to the global_bins_ map.
        std::lock_guard<std::mutex> lock(global_map_mutex_);
        auto it = global_bins_.find(alloc_size);
        if (it == global_bins_.end()) {
            // If no GlobalBin exists for this size, create one.
            it = global_bins_.emplace(alloc_size, std::make_unique<GlobalBin>()).first;
        }
        return it->second.get();
    }

    // Thread-local pointer to the current thread's cache.
    // Initialized to nullptr and managed by get_thread_cache and ThreadCacheWrapper.
    thread_local ThreadCache* tls_cache_ptr = nullptr;

    /**
     * @brief Helper struct to manage the lifecycle of thread-local caches.
     *
     * This wrapper ensures that when a thread exits, its associated ThreadCache
     * is properly deallocated and any remaining memory blocks are returned to the
     * global memory pool, preventing memory leaks and ensuring proper resource management.
     */
    struct ThreadCacheWrapper {
        // A dummy function to ensure the tls_wrapper is "touched" and thus
        // its destructor is called when the thread exits.
        void touch() {}

        /**
         * @brief Destructor for ThreadCacheWrapper.
         *
         * This is automatically called when the thread associated with `tls_wrapper` exits.
         * It performs cleanup of the thread-local cache.
         */
        ~ThreadCacheWrapper() {
            if (tls_cache_ptr) {  // Check if a ThreadCache was actually created for this thread.
                MemoryPool& pool = MemoryPool::get_instance();

                // Step 1: Remove the current thread's cache from the global list of all caches.
                // This prevents `reset()` from trying to access a destructing cache.
                {
                    std::lock_guard<std::mutex> lock(pool.caches_mutex_);
                    pool.all_caches_.erase(tls_cache_ptr);
                }

                // Step 2: Push any remaining blocks from the thread-local cache back to the global bins.
                // This makes the memory available for other threads or future allocations.
                for (auto& [size, blocks] : tls_cache_ptr->blocks) {
                    if (blocks.empty()) continue;  // Skip if no blocks of this size.
                    GlobalBin* bin = pool.get_global_bin(size);
                    std::lock_guard<std::mutex> bin_lock(bin->mutex);  // Protects access to the GlobalBin.
                    bin->blocks.insert(bin->blocks.end(), blocks.begin(), blocks.end());
                }

                // Step 3: Deallocate the ThreadCache object itself.
                delete tls_cache_ptr;
                // Critical: Set to nullptr to avoid use-after-free issues if the thread_local variable
                // somehow gets re-accessed after destruction, or if another part of the system
                // incorrectly assumes its validity.
                tls_cache_ptr = nullptr;
            }
        }
    };

    // Instantiate the thread-local wrapper. Its destructor will be called on thread exit.
    thread_local ThreadCacheWrapper tls_wrapper;

    /**
     * @brief Retrieves the ThreadCache for the current thread.
     *
     * If no ThreadCache exists for the current thread, one is created and registered
     * with the global MemoryPool.
     *
     * @return A pointer to the current thread's ThreadCache.
     */
    ThreadCache* MemoryPool::get_thread_cache() {
        if (!tls_cache_ptr) {                                 // Check if the current thread already has a cache.
            tls_cache_ptr = new ThreadCache();                // Create a new cache if not.
            std::lock_guard<std::mutex> lock(caches_mutex_);  // Protects access to all_caches_ set.
            all_caches_.insert(tls_cache_ptr);                // Register the new cache with the MemoryPool.
        }
        tls_wrapper.touch();  // Ensure the thread-local wrapper's destructor is called on thread exit.
        return tls_cache_ptr;
    }

    /**
     * @brief Allocates a block of memory of the specified size.
     *
     * This function attempts to allocate memory using a multi-level caching strategy:
     * 1. Try to get from the current thread's local cache (fastest).
     * 2. If local cache is empty, try to get from the global pool (shared, slower but avoids OS call).
     * 3. If global pool is also empty, allocate directly from the operating system.
     *
     * Memory is always allocated with 32-byte alignment to optimize for SIMD operations (e.g., AVX2/AVX-512).
     *
     * @param bytes The requested size of memory in bytes.
     * @return A pointer to the allocated memory block.
     * @throws std::bad_alloc if memory allocation from the OS fails.
     */
    void* MemoryPool::allocate(const size_t bytes) {
        if (bytes == 0) return nullptr;  // Cannot allocate zero bytes.

        // Calculate the actual allocation size, rounded up to the nearest 32-byte boundary.
        // This ensures memory is aligned for SIMD instructions.
        size_t alloc_size = (bytes + 31) & ~31;

        // Get the current thread's local cache.
        ThreadCache* local_cache = get_thread_cache();
        // Get the list of blocks for the specific alloc_size within the local cache.
        auto& local_list = local_cache->blocks[alloc_size];

        // --- Fast-path: Allocate from thread-local cache ---
        if (!local_list.empty()) {
            void* ptr = local_list.back();  // Take the last available block.
            local_list.pop_back();          // Remove it from the list.
            return ptr;
        }

        // --- Slow-path: Thread-local cache is empty, fetch from global pool ---
        GlobalBin* bin = get_global_bin(alloc_size);  // Get the global bin for this allocation size.
        {
            std::lock_guard<std::mutex> lock(bin->mutex);  // Protects access to the global bin's blocks.
            if (!bin->blocks.empty()) {
                // Transfer a batch of blocks from the global bin to the local cache.
                // This reduces contention on the global mutex.
                size_t transfer_count = std::min(TRANSFER_BATCH_SIZE, bin->blocks.size());
                auto transfer_start = bin->blocks.end() - transfer_count;

                // The last block in the batch is returned directly to the caller.
                void* ptr = *(bin->blocks.end() - 1);

                // If more than one block was transferred, put the rest into the local cache.
                if (transfer_count > 1) {
                    local_list.insert(local_list.end(), transfer_start, bin->blocks.end() - 1);
                }
                // Erase the transferred blocks from the global bin.
                bin->blocks.erase(transfer_start, bin->blocks.end());

                return ptr;
            }
        }

        // --- Fallback: Global pool is also empty, allocate directly from the OS ---
#if defined(_WIN32)
        // Use Windows-specific aligned allocation.
        void* ptr = _aligned_malloc(alloc_size, 32);
#else
        // Use standard C++17 aligned allocation or POSIX aligned_alloc.
        void* ptr = std::aligned_alloc(32, alloc_size);
#endif

        // Check for allocation failure.
        if (!ptr) throw std::bad_alloc();

        g_total_allocated.fetch_add(alloc_size, std::memory_order_relaxed);
        return ptr;
    }

    /**
     * @brief Deallocates a previously allocated block of memory.
     *
     * This function returns the memory block to the appropriate cache:
     * 1. If the current thread's cache is active, return to the local cache.
     * 2. If the thread is exiting or has no active cache, return directly to the global pool.
     *
     * The thread-local cache performs garbage collection, returning excess blocks
     * to the global pool when it grows too large.
     *
     * @param ptr A pointer to the memory block to deallocate.
     * @param bytes The original size of the memory block (used to determine `alloc_size`).
     */
    void MemoryPool::deallocate(void* ptr, const size_t bytes) {
        if (!ptr) return;  // Cannot deallocate a nullptr.

        // Calculate the standardized allocation size, matching the allocate function.
        const size_t alloc_size = (bytes + 31) & ~31;

        g_total_allocated.fetch_sub(alloc_size, std::memory_order_relaxed);

        // If tls_cache_ptr is nullptr, it means the thread is exiting or has never
        // allocated memory via this allocator. In this case, bypass the thread cache
        // and return the block directly to the global bin.
        if (!tls_cache_ptr) {
            GlobalBin* bin = get_global_bin(alloc_size);
            std::lock_guard<std::mutex> lock(bin->mutex);  // Protects access to the GlobalBin.
            bin->blocks.push_back(ptr);                    // Return the block to the global pool.
            return;
        }

        // Return the block to the current thread's local cache.
        ThreadCache* local_cache = tls_cache_ptr;
        auto& local_list = local_cache->blocks[alloc_size];
        local_list.push_back(ptr);

        // --- Garbage collection for thread-local cache ---
        // If the local cache for this size grows too large, transfer half of its blocks
        // back to the global pool. This helps balance memory distribution and prevents
        // a single thread from hoarding too much memory.
        if (local_list.size() >= MAX_LOCAL_CACHE_SIZE) {
            size_t transfer_count = MAX_LOCAL_CACHE_SIZE / 2;
            GlobalBin* bin = get_global_bin(alloc_size);

            std::lock_guard<std::mutex> lock(bin->mutex);  // Protects access to the GlobalBin.
            auto transfer_start = local_list.end() - transfer_count;
            bin->blocks.insert(bin->blocks.end(), transfer_start, local_list.end());  // Move blocks.
            local_list.erase(transfer_start, local_list.end());                       // Remove blocks from local cache.
        }
    }

    /**
     * @brief Resets the memory pool, freeing all currently held memory.
     *
     * This function iterates through all thread-local caches and global bins,
     * freeing all memory blocks back to the operating system.
     * It's called by the MemoryPool destructor and can be used for explicit cleanup.
     */
    void MemoryPool::reset() {
        // Step 1: Clear all thread-local caches.
        {
            std::lock_guard<std::mutex> lock(caches_mutex_);  // Protects access to the all_caches_ set.
            for (ThreadCache* cache : all_caches_) {
                for (auto& [size, blocks] : cache->blocks) {
                    for (void* ptr : blocks) {
#if defined(_WIN32)
                        _aligned_free(ptr);  // Windows-specific aligned free.
#else
                        std::free(ptr);  // Standard free for aligned_alloc.
#endif
                    }
                    blocks.clear();  // Clear the list of blocks for this size.
                }
            }
        }

        // Step 2: Clear all global bins.
        {
            std::lock_guard<std::mutex> lock(global_map_mutex_);  // Protects access to the global_bins_ map.
            for (auto& [size, bin_ptr] : global_bins_) {
                std::lock_guard<std::mutex> bin_lock(bin_ptr->mutex);  // Protects access to individual GlobalBin.
                for (void* ptr : bin_ptr->blocks) {
#if defined(_WIN32)
                    _aligned_free(ptr);
#else
                    std::free(ptr);
#endif
                }
                bin_ptr->blocks.clear();  // Clear the list of blocks.
            }
            global_bins_.clear();  // Clear the map of global bins.
        }
    }

    /**
     * @brief Provides a singleton instance of the MemoryPool.
     *
     * This ensures that there is only one global MemoryPool managing all memory
     * allocations and deallocations within the application.
     *
     * @return A reference to the single MemoryPool instance.
     */
    MemoryPool& MemoryPool::get_instance() {
        // Meyers' singleton: thread-safe initialization on first call.
        static MemoryPool instance;
        return instance;
    }

    /**
     * @brief Returns the number of blocks currently in the global pool for a given size.
     *
     * @param alloc_size The standardized size of the memory blocks to query.
     * @return The number of cached memory blocks of that size.
     */
    size_t MemoryPool::get_global_pool_size(size_t alloc_size) {
        GlobalBin* bin = get_global_bin(alloc_size);
        std::lock_guard<std::mutex> lock(bin->mutex);
        return bin->blocks.size();
    }

}  // namespace helix
