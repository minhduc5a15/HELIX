#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "nn/linear.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"

using namespace helix;

class SequentialStressTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(SequentialStressTest, MemoryLeakStressTest) {
    for (int i = 0; i < 5000; ++i) {
        Sequential model(Linear(4, 8), ReLU(), Linear(8, 2));
        Tensor x = Tensor::randn({32, 4});
        Tensor y = model(x);
        Tensor loss = y.sum();
        loss.backward();
    }
    SUCCEED();
}

TEST_F(SequentialStressTest, DeepGraphTeardownStressTest) {
    // Tests teardown of a very deep Sequential network to catch stack overflow in Graph teardown
    std::vector<std::shared_ptr<Module>> layers;
    layers.push_back(std::make_shared<Linear>(2, 4));
    for (int i = 0; i < 500; ++i) {
        layers.push_back(std::make_shared<ReLU>());
        layers.push_back(std::make_shared<Linear>(4, 4));
    }
    layers.push_back(std::make_shared<Linear>(4, 1));

    Sequential model(layers);

    Tensor x = Tensor::randn({1, 2});
    Tensor y = model(x);
    Tensor loss = y.sum();
    loss.backward();

    EXPECT_EQ(model.parameters()[0].grad().shape().vec(), (std::vector<size_t>{2, 4}));
}
