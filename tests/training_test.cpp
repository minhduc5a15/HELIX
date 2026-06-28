#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "nn/linear.hpp"
#include "nn/loss.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"
#include "optim/sgd.hpp"

using namespace helix;

class TrainingTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(TrainingTest, MiniTrainingLoop) {
    // Train a simple MLP: Linear(2, 4) -> ReLU() -> Linear(4, 1)
    // on a small batch of dummy data to check if loss decreases.
    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));

    SGD optimizer(model.parameters(), 0.05f);

    // Inputs: size [4, 2]
    Tensor x = Tensor::randn({4, 2});
    // Target: size [4, 1]
    Tensor target = Tensor::randn({4, 1});

    // Initial evaluation
    Tensor y_init = model(x);
    Tensor loss_init = mse_loss(y_init, target);
    float initial_loss_val = loss_init.item();

    // Run training for 15 epochs
    float last_loss = initial_loss_val;
    for (int epoch = 0; epoch < 15; ++epoch) {
        optimizer.zero_grad();

        Tensor y = model(x);
        Tensor loss = mse_loss(y, target);
        last_loss = loss.item();

        loss.backward();
        optimizer.step();
    }

    // Verify that the final loss is significantly less than initial loss
    EXPECT_LT(last_loss, initial_loss_val);
}
