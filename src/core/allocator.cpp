#include "core/allocator.hpp"

#include <cstdlib>
#include <new>
#include <algorithm>

namespace helix {

    MemoryPool::MemoryPool() = default;
    
    MemoryPool::~MemoryPool() { 
        reset(); 
    }

    GlobalBin* MemoryPool::get_global_bin(size_t alloc_size) {
        std::lock_guard<std::mutex> lock(global_map_mutex_);
        auto it = global_bins_.find(alloc_size);
        if (it == global_bins_.end()) {
            it = global_bins_.emplace(alloc_size, std::make_unique<GlobalBin>()).first;
        }
        return it->second.get();
    }

    struct ThreadCacheWrapper {
        ThreadCache* cache;
        MemoryPool* pool;

        ThreadCacheWrapper(MemoryPool* p) : pool(p) {
            cache = new ThreadCache();
            std::lock_guard<std::mutex> lock(pool->caches_mutex_);
            pool->all_caches_.insert(cache);
        }

        ~ThreadCacheWrapper() {
            // First remove from global list to avoid reset() accessing it while destructing
            {
                std::lock_guard<std::mutex> lock(pool->caches_mutex_);
                pool->all_caches_.erase(cache);
            }
            
            // Push remaining blocks to global bins
            for (auto& [size, blocks] : cache->blocks) {
                if (blocks.empty()) continue;
                GlobalBin* bin = pool->get_global_bin(size);
                std::lock_guard<std::mutex> bin_lock(bin->mutex);
                bin->blocks.insert(bin->blocks.end(), blocks.begin(), blocks.end());
            }
            delete cache;
        }
    };

    ThreadCache* MemoryPool::get_thread_cache() {
        thread_local ThreadCacheWrapper wrapper(this);
        return wrapper.cache;
    }

    void* MemoryPool::allocate(const size_t bytes) {
        if (bytes == 0) return nullptr;

        // 32-byte alignment for SIMD operations (AVX2/AVX-512)
        size_t alloc_size = (bytes + 31) & ~31;

        ThreadCache* local_cache = get_thread_cache();
        auto& local_list = local_cache->blocks[alloc_size];

        // Fast-path: take from local cache
        if (!local_list.empty()) {
            void* ptr = local_list.back();
            local_list.pop_back();
            return ptr;
        }

        // Slow-path: fetch from global pool
        GlobalBin* bin = get_global_bin(alloc_size);
        {
            std::lock_guard<std::mutex> lock(bin->mutex);
            if (!bin->blocks.empty()) {
                // Transfer a batch of blocks from global to local
                size_t transfer_count = std::min(TRANSFER_BATCH_SIZE, bin->blocks.size());
                auto transfer_start = bin->blocks.end() - transfer_count;
                
                // Keep the last one to return directly, put the rest in local cache
                void* ptr = *(bin->blocks.end() - 1);
                
                if (transfer_count > 1) {
                    local_list.insert(local_list.end(), transfer_start, bin->blocks.end() - 1);
                }
                bin->blocks.erase(transfer_start, bin->blocks.end());
                
                return ptr;
            }
        }

        // Global pool also empty, allocate from OS
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

        ThreadCache* local_cache = get_thread_cache();
        auto& local_list = local_cache->blocks[alloc_size];

        local_list.push_back(ptr);

        // Garbage collection: if local cache is too large, transfer half back to global pool
        if (local_list.size() >= MAX_LOCAL_CACHE_SIZE) {
            size_t transfer_count = MAX_LOCAL_CACHE_SIZE / 2;
            GlobalBin* bin = get_global_bin(alloc_size);
            
            std::lock_guard<std::mutex> lock(bin->mutex);
            auto transfer_start = local_list.end() - transfer_count;
            bin->blocks.insert(bin->blocks.end(), transfer_start, local_list.end());
            local_list.erase(transfer_start, local_list.end());
        }
    }

    void MemoryPool::reset() {
        // Clear all thread-local caches
        {
            std::lock_guard<std::mutex> lock(caches_mutex_);
            for (ThreadCache* cache : all_caches_) {
                for (auto& [size, blocks] : cache->blocks) {
                    for (void* ptr : blocks) {
#if defined(_WIN32)
                        _aligned_free(ptr);
#else
                        std::free(ptr);
#endif
                    }
                    blocks.clear();
                }
            }
        }

        // Clear global bins
        {
            std::lock_guard<std::mutex> lock(global_map_mutex_);
            for (auto& [size, bin_ptr] : global_bins_) {
                std::lock_guard<std::mutex> bin_lock(bin_ptr->mutex);
                for (void* ptr : bin_ptr->blocks) {
#if defined(_WIN32)
                    _aligned_free(ptr);
#else
                    std::free(ptr);
#endif
                }
                bin_ptr->blocks.clear();
            }
        }
    }

    MemoryPool& MemoryPool::get_instance() {
        static MemoryPool instance;
        return instance;
    }

}  // namespace helix
