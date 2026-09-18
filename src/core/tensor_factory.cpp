#include "core/tensor_factory.hpp"

#include <random>

#include "core/dispatcher.hpp"

namespace helix {

    Tensor TensorFactory::empty(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) {
        DType dt = dtype.value_or(DType::Float32);
        Device dev = device.value_or(Device(DeviceType::CPU));
        return Tensor(shape, dt, dev);
    }

    Tensor TensorFactory::zeros(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) {
        return full(shape, 0.0f, dtype, device);
    }

    Tensor TensorFactory::ones(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) {
        return full(shape, 1.0f, dtype, device);
    }

    Tensor TensorFactory::full(
        const Shape& shape, const float value, std::optional<DType> dtype, std::optional<Device> device
    ) {
        DType dt = dtype.value_or(DType::Float32);
        Device dev = device.value_or(Device(DeviceType::CPU));
        Tensor t(shape, dt, dev);
        const size_t n = t.numel();
        HELIX_DISPATCH_ALL_TYPES(dt, "TensorFactory::full", [&] {
            scalar_t* data = t.data_ptr<scalar_t>();
            scalar_t cast_value = static_cast<scalar_t>(value);
            for (size_t i = 0; i < n; ++i) {
                data[i] = cast_value;
            }
        });
        return t;
    }

    Tensor TensorFactory::randn(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) {
        DType dt = dtype.value_or(DType::Float32);
        Device dev = device.value_or(Device(DeviceType::CPU));
        Tensor t(shape, dt, dev);
        const size_t n = t.numel();

        // Use thread_local generator to eliminate cross-thread data races.
        thread_local std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        HELIX_DISPATCH_ALL_TYPES(dt, "TensorFactory::randn", [&] {
            scalar_t* data = t.data_ptr<scalar_t>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<scalar_t>(dist(gen));
            }
        });
        return t;
    }

    Tensor cat(const std::vector<Tensor>& tensors, size_t dim) { return Dispatcher::cat(tensors, dim); }

}  // namespace helix
