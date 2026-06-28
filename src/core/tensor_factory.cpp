#include "core/tensor_factory.hpp"

#include <random>

namespace helix {

    Tensor TensorFactory::empty(const Shape& shape) {
        // In the current implementation, Tensor(shape) will allocate and default-initialize (zero) memory via
        // std::vector. For true "empty" (uninitialized) we would need to bypass std::vector initialization in
        // Allocator, but for now this is perfectly safe and functional.
        return Tensor(shape);
    }

    Tensor TensorFactory::zeros(const Shape& shape) { return full(shape, 0.0f); }

    Tensor TensorFactory::ones(const Shape& shape) { return full(shape, 1.0f); }

    Tensor TensorFactory::full(const Shape& shape, const float value) {
        Tensor t(shape);
        float* data = t.data_ptr();
        const size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) {
            data[i] = value;
        }
        return t;
    }

    Tensor TensorFactory::randn(const Shape& shape) {
        Tensor t(shape);
        float* data = t.data_ptr();
        const size_t n = t.numel();

        // Use a fixed seed for reproducible tests, or random_device for true randomness.
        static std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
        return t;
    }

}  // namespace helix
