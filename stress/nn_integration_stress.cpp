#include <gtest/gtest.h>
#include "autograd/engine.hpp"
#include "nn/linear.hpp"
#include "nn/loss.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"

using namespace helix;

class NNIntegrationStressTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(NNIntegrationStressTest, ResourceAllocation_MemoryLeakStressTest) {
    for (int i = 0; i < 5000; ++i) {
        Sequential net(Linear(2, 32), ReLU(), Linear(32, 16), ReLU(), Linear(16, 1));
        Tensor x = Tensor::randn({128, 2});
        Tensor target = Tensor::randn({128, 1});
        Tensor y = net(x);
        Tensor loss = mse_loss(y, target);
        loss.backward();
    }
    SUCCEED();
}
