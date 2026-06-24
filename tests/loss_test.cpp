#include "nn/loss.hpp"

#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "grad_check.hpp"

using namespace helix;

class LossTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(LossTest, MSE) {
    Tensor pred({2});
    pred.data_ptr()[0] = 1.0f;
    pred.data_ptr()[1] = 2.0f;
    pred.set_requires_grad(true);

    Tensor target({2});
    target.data_ptr()[0] = 1.0f;
    target.data_ptr()[1] = 3.0f;

    Tensor loss = mse_loss(pred, target);

    EXPECT_FLOAT_EQ(loss.item(), 0.5f);

    loss.backward();

    Tensor grad = pred.grad();
    EXPECT_EQ(grad.shape().vec(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(grad.data_ptr()[0], 0.0f);
    EXPECT_FLOAT_EQ(grad.data_ptr()[1], -1.0f);
}

TEST_F(LossTest, GradientCheck) {
    Tensor pred = Tensor::randn({4, 4});
    Tensor target = Tensor::randn({4, 4});

    auto func = [](const std::vector<Tensor>& args) { return mse_loss(args[0], args[1]); };

    EXPECT_TRUE(gradient_check(func, {pred, target}));
}

TEST_F(LossTest, ErrorHandling_MismatchShapeNoBroadcast) {
    Tensor pred = Tensor::randn({32, 5});
    Tensor target = Tensor::randn({32, 4});
    EXPECT_THROW({ mse_loss(pred, target); }, std::exception);
}

TEST_F(LossTest, EdgeCases_ZeroLoss) {
    Tensor target = Tensor::zeros({4, 4});
    Tensor pred = Tensor::zeros({4, 4});
    pred.set_requires_grad(true);

    Tensor loss = mse_loss(pred, target);
    EXPECT_FLOAT_EQ(loss.item(), 0.0f);

    loss.backward();

    Tensor grad = pred.grad();
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(grad.data_ptr()[i], 0.0f);
    }
}

TEST_F(LossTest, Broadcasting_TargetBroadcastable) {
    Tensor pred = Tensor::randn({32, 4});
    pred.set_requires_grad(true);
    Tensor target = Tensor::randn({4});  // Shape {4} broadcasts to {32, 4}

    Tensor loss = mse_loss(pred, target);
    loss.backward();

    EXPECT_EQ(pred.grad().shape().vec(), (std::vector<size_t>{32, 4}));
}

TEST_F(LossTest, Contiguity_NonContiguousPredAndTarget) {
    Tensor pred = Tensor::randn({4, 4});
    Tensor pred_t = pred.transpose(0, 1);
    pred_t.set_requires_grad(true);

    Tensor target = Tensor::randn({4, 4});
    Tensor target_t = target.transpose(0, 1);

    Tensor loss = mse_loss(pred_t, target_t);
    loss.backward();

    EXPECT_EQ(pred_t.grad().shape().vec(), (std::vector<size_t>{4, 4}));
}
