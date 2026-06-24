#pragma once

#include <memory>

#include "core/autograd_meta.hpp"
#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/shape.hpp"
#include "core/storage.hpp"
#include "core/stride.hpp"

namespace helix {

    class TensorImpl {
    public:
        TensorImpl(Shape shape, const DType dtype = DType::Float32, const Device device = Device(DeviceType::CPU))
            : shape_(std::move(shape)),
              stride_(Stride::compute_contiguous(shape_)),
              dtype_(dtype),
              device_(device),
              storage_offset_(0) {
            size_t bytes = shape_.numel() * dtype_size(dtype_);
            storage_ = std::make_shared<Storage>(bytes);
        }

        // Constructor for views
        TensorImpl(
            std::shared_ptr<Storage> storage,
            const size_t offset,
            Shape shape,
            Stride stride,
            DType dtype,
            Device device
        )
            : shape_(std::move(shape)),
              stride_(std::move(stride)),
              dtype_(dtype),
              device_(device),
              storage_(std::move(storage)),
              storage_offset_(offset) {}

        const Shape& shape() const { return shape_; }
        const Stride& stride() const { return stride_; }
        DType dtype() const { return dtype_; }
        Device device() const { return device_; }
        size_t storage_offset() const { return storage_offset_; }
        const std::shared_ptr<Storage>& storage() const { return storage_; }

        bool is_contiguous() const { return stride_ == Stride::compute_contiguous(shape_); }

        // Pointer to the first element
        void* data() {
            if (!storage_ || !storage_->data()) return nullptr;
            return static_cast<uint8_t*>(storage_->data()) + (storage_offset_ * dtype_size(dtype_));
        }

        const void* data() const {
            if (!storage_ || !storage_->data()) return nullptr;
            return static_cast<const uint8_t*>(storage_->data()) + (storage_offset_ * dtype_size(dtype_));
        }

        AutogradMeta* autograd_meta() const { return autograd_meta_.get(); }
        void set_autograd_meta(std::unique_ptr<AutogradMeta, AutogradMetaDeleter> meta) {
            autograd_meta_ = std::move(meta);
        }

    private:
        Shape shape_;
        Stride stride_;
        DType dtype_;
        Device device_;

        std::shared_ptr<Storage> storage_;
        size_t storage_offset_;  // Offset in elements (not bytes)

        std::unique_ptr<AutogradMeta, AutogradMetaDeleter> autograd_meta_;
    };

}  // namespace helix
