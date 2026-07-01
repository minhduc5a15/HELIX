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

TEST_F(TrainingTest, LinearRegression_Convergence) {
    // y = 2x + 1
    Tensor x = Tensor::randn({100, 1});
    Tensor target = x * 2.0f + 1.0f;

    Sequential model(Linear(1, 1));
    SGD optimizer(model.parameters(), 0.01f);

    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 500; ++epoch) {
        optimizer.zero_grad();
        Tensor y = model(x);
        Tensor loss = mse_loss(y, target);
        last_loss = loss.item();
        loss.backward();
        optimizer.step();
    }

    // Tiêu chí: Loss giảm xuống dưới một ngưỡng xác định
    EXPECT_LT(last_loss, 1e-1f);

    // Tiêu chí: Tham số học được xấp xỉ nghiệm kỳ vọng
    auto params = model.parameters();
    // params[0] is Weight, params[1] is Bias
    EXPECT_NEAR(params[0].item(), 2.0f, 0.2f);
    EXPECT_NEAR(params[1].item(), 1.0f, 0.2f);
}

TEST_F(TrainingTest, XOR_Convergence) {
    Tensor x({0.0f, 0.0f, 
              0.0f, 1.0f, 
              1.0f, 0.0f, 
              1.0f, 1.0f}, Shape{4, 2});
    Tensor target({0.0f, 1.0f, 1.0f, 0.0f}, Shape{4, 1});

    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));
    SGD optimizer(model.parameters(), 0.1f);

    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 1000; ++epoch) {
        optimizer.zero_grad();
        Tensor y = model(x);
        Tensor loss = mse_loss(y, target);
        last_loss = loss.item();
        loss.backward();
        optimizer.step();
    }

    // Tiêu chí: Loss hội tụ về mức rất nhỏ
    EXPECT_LT(last_loss, 0.1f);

    // Tiêu chí: Prediction error < tolerance
    Tensor pred = model(x);
    const float* pred_data = pred.data_ptr();
    const float* target_data = target.data_ptr();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(pred_data[i], target_data[i], 0.3f);
    }
}

