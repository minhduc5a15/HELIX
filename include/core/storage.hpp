#pragma once

#include "core/allocator.hpp"

namespace helix {

    // Storage manages the raw linear memory buffer for Tensors.
    // It uses Reference Counting (via std::shared_ptr externally) so that multiple Tensors
    // can share the same underlying data (e.g., in Tensor Views).
    class Storage {
    public:
        // Create storage with given size (in bytes), allocating from the specified allocator.
        // Defaults to the global MemoryPool if allocator is nullptr.
        explicit Storage(size_t size_bytes, Allocator* allocator = nullptr);

        ~Storage();

        // Non-copyable to prevent accidental deep copies of raw pointers
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;

        // Movable
        Storage(Storage&& other) noexcept;
        Storage& operator=(Storage&& other) noexcept;

        void* data() { return data_; }
        const void* data() const { return data_; }
        size_t size_bytes() const { return size_bytes_; }

    private:
        void* data_{nullptr};
        size_t size_bytes_{0};
        Allocator* allocator_{nullptr};
    };

}  // namespace helix
