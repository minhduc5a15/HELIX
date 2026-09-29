#include "core/tensor_factory.hpp"

#include <chrono>
#include <random>
#include <thread>

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

    namespace {
        std::atomic<bool> g_has_manual_seed{false};
        std::atomic<uint64_t> g_global_seed{0};

        thread_local std::mt19937 tl_gen([] {
            if (g_has_manual_seed.load(std::memory_order_relaxed)) {
                const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
                return std::mt19937(
                    static_cast<std::mt19937::result_type>(g_global_seed.load(std::memory_order_relaxed) + tid)
                );
            }
            std::random_device rd;
            const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
            const auto ts =
                static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            return std::mt19937(rd() ^ static_cast<unsigned int>(tid) ^ ts);
        }());
    }  // namespace

    void TensorFactory::manual_seed(const uint64_t seed) {
        g_global_seed.store(seed, std::memory_order_relaxed);
        g_has_manual_seed.store(true, std::memory_order_relaxed);
        tl_gen.seed(static_cast<std::mt19937::result_type>(seed));
    }

    void manual_seed(const uint64_t seed) { TensorFactory::manual_seed(seed); }

    Tensor TensorFactory::randn(const Shape& shape, std::optional<DType> dtype, std::optional<Device> device) {
        DType dt = dtype.value_or(DType::Float32);
        Device dev = device.value_or(Device(DeviceType::CPU));
        Tensor t(shape, dt, dev);
        const size_t n = t.numel();

        std::normal_distribution<float> dist(0.0f, 1.0f);

        HELIX_DISPATCH_ALL_TYPES(dt, "TensorFactory::randn", [&] {
            scalar_t* data = t.data_ptr<scalar_t>();
            for (size_t i = 0; i < n; ++i) {
                data[i] = static_cast<scalar_t>(dist(tl_gen));
            }
        });
        return t;
    }

    Tensor cat(const std::vector<Tensor>& tensors, size_t dim) { return Dispatcher::cat(tensors, dim); }

}  // namespace helix
