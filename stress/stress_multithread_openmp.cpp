#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "core/tensor.hpp"

using namespace helix;

TEST(StressMultithread, OpenMPMemoryPoolConcurrency) {
    const size_t num_threads = 16;
    std::vector<std::thread> threads;

    auto worker = []() {
        for (int i = 0; i < 100; ++i) {
            // Allocate and deallocate tensors rapidly
            Tensor a = Tensor::randn({128, 128});
            Tensor b = Tensor::randn({128, 128});

            // This triggers OpenMP matmul internally if sizes are large enough
            Tensor c = a.matmul(b);

            // Trigger broadcast sum to test strides
            Tensor d = c.sum(0, true);

            // Trigger in-place modification
            Tensor e = d + 1.0f;
            e.add_(d);
        }
    };

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }

    // If it survives without segfaults or TSan errors, it passes.
    SUCCEED();
}
