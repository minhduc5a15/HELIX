#include <gtest/gtest.h>
#include "nn/linear.hpp"
#include "core/tensor.hpp"
#include "autograd/engine.hpp"

using namespace helix;

class LinearStressTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(LinearStressTest, MemoryLeakStressTest) {
    // Sustained small allocations to catch memory leaks
    Linear fc(4, 8);
    for (int i = 0; i < 5000; ++i) {
        Tensor x = Tensor::randn({32, 4});
        Tensor y = fc(x);
        Tensor loss = y.sum();
        loss.backward();
    }
    SUCCEED();
}

TEST_F(LinearStressTest, HighVolumeDataStressTest) {
    // Large matrix multiplication to test memory limits and contiguity handling under pressure
    Linear fc(512, 1024);
    for (int i = 0; i < 10; ++i) {
        Tensor x = Tensor::randn({256, 512});
        Tensor y = fc(x);
        Tensor loss = y.sum();
        loss.backward();
        
        EXPECT_EQ(fc.parameters()[0].grad().shape().vec(), (std::vector<size_t>{512, 1024}));
    }
    SUCCEED();
}
