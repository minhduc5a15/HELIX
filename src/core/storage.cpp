#include "core/storage.hpp"

namespace helix {

    Storage::Storage(size_t size_bytes, Allocator* allocator) : size_bytes_(size_bytes), allocator_(allocator) {
        if (!allocator_) {
            allocator_ = &MemoryPool::global();
        }
        if (size_bytes_ > 0) {
            data_ = allocator_->allocate(size_bytes_);
        }
    }

    Storage::~Storage() {
        if (data_ && allocator_) {
            allocator_->deallocate(data_, size_bytes_);
        }
    }

    Storage::Storage(Storage&& other) noexcept
        : data_(other.data_), size_bytes_(other.size_bytes_), allocator_(other.allocator_) {
        other.data_ = nullptr;
        other.size_bytes_ = 0;
    }

    Storage& Storage::operator=(Storage&& other) noexcept {
        if (this != &other) {
            if (data_ && allocator_) {
                allocator_->deallocate(data_, size_bytes_);
            }
            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            allocator_ = other.allocator_;

            other.data_ = nullptr;
            other.size_bytes_ = 0;
        }
        return *this;
    }

}  // namespace helix
