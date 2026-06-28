#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "nn/relu.hpp"

using namespace helix;

class ReLUStressTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(ReLUStressTest, MemoryLeakStressTest) {
    // Tests sustained small allocations to ensure no memory bloat.
    for (int i = 0; i < 5000; ++i) {
        Tensor x = Tensor::randn({32, 4});
        x.set_requires_grad(true);
        Tensor y = relu(x);
        Tensor loss = y.sum();
        loss.backward();
    }
    SUCCEED();
}

TEST_F(ReLUStressTest, HighVolumeDataStressTest) {
    // Tests a very large allocation to ensure memory pool can handle large blocks,
    // and that computations (ops) don't overflow or corrupt memory.
    for (int i = 0; i < 10; ++i) {
        Tensor x = Tensor::randn({1024, 1024});
        x.set_requires_grad(true);
        Tensor y = relu(x);
        Tensor loss = y.sum();
        loss.backward();

        EXPECT_EQ(x.grad().shape().vec(), (std::vector<size_t>{1024, 1024}));
    }
    SUCCEED();
}

TEST_F(ReLUStressTest, DeepGraphTeardownStressTest) {
    // Tests if tearing down a very deep computational graph causes stack overflow.
    // std::shared_ptr destruction is recursive, so a deep chain can overflow the stack.
    Tensor x = Tensor::randn({10});
    x.set_requires_grad(true);

    Tensor current = x;
    for (int i = 0; i < 1000; ++i) {
        current = relu(current);
    }

    Tensor loss = current.sum();
    // This triggers 1000-deep backward recursion, and when it falls out of scope,
    // a 1000-deep shared_ptr destruction recursion.
    loss.backward();

    EXPECT_EQ(x.grad().shape().vec(), (std::vector<size_t>{10}));
}
