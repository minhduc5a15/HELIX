#include "nn/loss.hpp"

#include <gtest/gtest.h>

#include <cmath>

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

// =========================================================================
// CrossEntropyLoss Tests
// =========================================================================

TEST_F(LossTest, CrossEntropy_ValueCheck) {
    // 2 samples, 3 classes
    Tensor pred({2, 3});
    // Sample 0 logits: 1.0, 2.0, 3.0
    pred.data_ptr()[0] = 1.0f;
    pred.data_ptr()[1] = 2.0f;
    pred.data_ptr()[2] = 3.0f;
    // Sample 1 logits: -1.0, 0.0, 1.0
    pred.data_ptr()[3] = -1.0f;
    pred.data_ptr()[4] = 0.0f;
    pred.data_ptr()[5] = 1.0f;
    pred.set_requires_grad(true);

    // Target probabilities
    Tensor target({2, 3});
    target.data_ptr()[0] = 0.0f;
    target.data_ptr()[1] = 0.0f;
    target.data_ptr()[2] = 1.0f;

    target.data_ptr()[3] = 0.0f;
    target.data_ptr()[4] = 1.0f;
    target.data_ptr()[5] = 0.0f;

    // Manual calc for Sample 0:
    // exp(1), exp(2), exp(3) -> max is 3
    // shifted: -2, -1, 0 -> exp(-2), exp(-1), 1 -> sum = 0.1353 + 0.3678 + 1 = 1.5032
    // log_softmax for class 2 = 0 - log(1.5032) = -0.4076

    // Manual calc for Sample 1:
    // exp(-1), exp(0), exp(1) -> max is 1
    // shifted: -2, -1, 0 -> exp(-2), exp(-1), 1 -> sum = 1.5032
    // log_softmax for class 1 = -1 - log(1.5032) = -1.4076

    // Mean loss = (0.4076 + 1.4076) / 2 = 0.9076

    Tensor loss = cross_entropy_loss(pred, target);
    EXPECT_NEAR(loss.item(), 0.9076f, 1e-3);

    loss.backward();

    Tensor grad = pred.grad();
    EXPECT_EQ(grad.shape().vec(), (std::vector<size_t>{2, 3}));

    // Grad check for sample 0, class 2
    // grad = (softmax_i - target_i) / N
    // softmax_2 = exp(-0.4076) = 0.6652
    // grad = (0.6652 - 1) / 2 = -0.1674
    EXPECT_NEAR(grad.data_ptr()[2], -0.1674f, 1e-3);
}

TEST_F(LossTest, CrossEntropy_GradientCheck) {
    Tensor pred = Tensor::randn({4, 5});  // 4 samples, 5 classes
    // One-hot encode targets for stability
    Tensor target = Tensor::zeros({4, 5});
    target.data_ptr()[0] = 1.0f;
    target.data_ptr()[6] = 1.0f;
    target.data_ptr()[12] = 1.0f;
    target.data_ptr()[18] = 1.0f;

    auto func = [target](const std::vector<Tensor>& args) { return cross_entropy_loss(args[0], target); };
    EXPECT_TRUE(gradient_check(func, {pred}));
}

TEST_F(LossTest, CrossEntropy_ErrorHandling_ShapeMismatch) {
    Tensor pred = Tensor::randn({32, 5});
    Tensor target = Tensor::randn({32, 4});
    EXPECT_THROW({ cross_entropy_loss(pred, target); }, std::exception);

    Tensor pred_1d = Tensor::randn({10});
    Tensor target_1d = Tensor::randn({10});
    EXPECT_THROW({ cross_entropy_loss(pred_1d, target_1d); }, std::exception);
}

TEST_F(LossTest, CrossEntropy_EdgeCases_ExtremeLogits) {
    // Test if log-sum-exp trick prevents NaN when exp(88) overflows
    Tensor pred({2, 2});
    pred.data_ptr()[0] = 1000.0f;
    pred.data_ptr()[1] = -1000.0f;
    pred.data_ptr()[2] = 0.0f;
    pred.data_ptr()[3] = 100.0f;
    pred.set_requires_grad(true);

    Tensor target({2, 2});
    target.data_ptr()[0] = 1.0f;
    target.data_ptr()[1] = 0.0f;
    target.data_ptr()[2] = 0.0f;
    target.data_ptr()[3] = 1.0f;

    Tensor loss = cross_entropy_loss(pred, target);
    // Should compute valid loss without NaN
    EXPECT_FALSE(std::isnan(loss.item()));
    EXPECT_FALSE(std::isinf(loss.item()));

    loss.backward();
    Tensor grad = pred.grad();
    EXPECT_FALSE(std::isnan(grad.data_ptr()[0]));
}

TEST_F(LossTest, CrossEntropy_Contiguity_NonContiguous) {
    Tensor pred = Tensor::randn({4, 3});
    Tensor pred_t = pred.transpose(0, 1);  // shape {3, 4}
    pred_t.set_requires_grad(true);

    Tensor target = Tensor::randn({4, 3});
    Tensor target_t = target.transpose(0, 1);  // shape {3, 4}

    Tensor loss = cross_entropy_loss(pred_t, target_t);
    loss.backward();

    EXPECT_EQ(pred_t.grad().shape().vec(), (std::vector<size_t>{3, 4}));
}
