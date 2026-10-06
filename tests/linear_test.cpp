#include "nn/linear.hpp"

#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/tensor.hpp"
#include "grad_check.hpp"

using namespace helix;

class LinearTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(LinearTest, ShapeTest) {
    Linear fc(4, 8);
    Tensor x = Tensor::randn({32, 4});
    Tensor y = fc(x);

    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{32, 8}));
}

TEST_F(LinearTest, ParametersCount) {
    Linear fc(4, 8);
    auto params = fc.parameters();

    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params[0].shape().vec(), (std::vector<size_t>{4, 8}));  // weight
    EXPECT_EQ(params[1].shape().vec(), (std::vector<size_t>{8}));     // bias
}

TEST_F(LinearTest, GradientCheck) {
    // Fixed deterministic inputs for cross-platform stability
    std::vector<float> x_data(2 * 4);
    for (size_t i = 0; i < x_data.size(); ++i) {
        x_data[i] = 0.1f * static_cast<float>(i + 1);
    }
    std::vector<float> w_data(4 * 8);
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = 0.05f * static_cast<float>(i + 1) - 0.8f;
    }
    std::vector<float> b_data(8);
    for (size_t i = 0; i < b_data.size(); ++i) {
        b_data[i] = 0.2f * static_cast<float>(i + 1);
    }

    Tensor x(x_data, Shape{2, 4});
    Tensor w(w_data, Shape{4, 8});
    Tensor b(b_data, Shape{8});

    auto func = [](const std::vector<Tensor>& args) { return (args[0].matmul(args[1]) + args[2]).sum(); };

    EXPECT_TRUE(gradient_check(func, {x, w, b}));
}

TEST_F(LinearTest, EdgeCases_BatchSizeOne) {
    Linear fc(4, 8);
    Tensor x = Tensor::randn({1, 4});
    Tensor y = fc(x);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{1, 8}));
}

TEST_F(LinearTest, EdgeCases_FeaturesOne) {
    Linear fc(1, 8);
    Tensor x = Tensor::randn({32, 1});
    Tensor y = fc(x);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{32, 8}));
}

TEST_F(LinearTest, ErrorHandling_InvalidShape) {
    Linear fc(4, 8);
    Tensor x = Tensor::randn({32, 5});  // mismatch
    EXPECT_THROW(
        { fc(x); }, std::invalid_argument
    );  // matmul throws std::invalid_argument for mismatched inner dimensions or we can use std::exception
}

TEST_F(LinearTest, Contiguity_NonContiguousInput) {
    Linear fc(4, 8);
    Tensor x = Tensor::randn({4, 4});
    Tensor x_t = x.transpose(0, 1);
    EXPECT_FALSE(x_t.is_contiguous());

    Tensor y = fc(x_t);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{4, 8}));

    Tensor loss = y.sum();
    loss.backward();
    EXPECT_EQ(fc.parameters()[0].grad().shape().vec(), (std::vector<size_t>{4, 8}));
}

TEST_F(LinearTest, LinearForward) {
    // Math verification on known constant weights and inputs
    // Input: shape [1, 2], values: [2.0, 3.0]
    // Weight: shape [2, 2], values: [1.0, 2.0, 3.0, 4.0]
    // Bias: shape [2], values: [0.5, 1.5]
    // Output should be: [2.0*1.0 + 3.0*3.0 + 0.5, 2.0*2.0 + 3.0*4.0 + 1.5]
    // = [11.5, 17.5]

    Linear fc(2, 2);

    Tensor w = Tensor::zeros({2, 2});
    w.data_ptr()[0] = 1.0f;
    w.data_ptr()[1] = 2.0f;
    w.data_ptr()[2] = 3.0f;
    w.data_ptr()[3] = 4.0f;

    Tensor b = Tensor::zeros({2});
    b.data_ptr()[0] = 0.5f;
    b.data_ptr()[1] = 1.5f;

    {
        no_grad guard;
        fc.parameters()[0].copy_(w);
        fc.parameters()[1].copy_(b);
    }

    Tensor x = Tensor::zeros({1, 2});
    x.data_ptr()[0] = 2.0f;
    x.data_ptr()[1] = 3.0f;

    Tensor y = fc(x);

    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{1, 2}));
    EXPECT_FLOAT_EQ(y.data_ptr()[0], 11.5f);
    EXPECT_FLOAT_EQ(y.data_ptr()[1], 17.5f);
}

TEST_F(LinearTest, LinearOutputShape) {
    // Explicitly test various shapes to verify output size
    {
        Linear fc(10, 5);
        Tensor x = Tensor::randn({1, 10});
        EXPECT_EQ(fc(x).shape().vec(), (std::vector<size_t>{1, 5}));
    }
    {
        Linear fc(10, 5);
        Tensor x = Tensor::randn({128, 10});
        EXPECT_EQ(fc(x).shape().vec(), (std::vector<size_t>{128, 5}));
    }
}

TEST_F(LinearTest, BiasBroadcasting) {
    // Explicitly check bias broadcasting correctness.
    // Weight set to Identity matrix, so X * W = X.
    // Input X of shape [3, 2] gets bias [1.0, 2.0] added to each of its 3 rows.
    Linear fc(2, 2);

    Tensor w = Tensor::zeros({2, 2});
    w.data_ptr()[0] = 1.0f;  // I
    w.data_ptr()[3] = 1.0f;

    Tensor b = Tensor::zeros({2});
    b.data_ptr()[0] = 1.0f;
    b.data_ptr()[1] = 2.0f;

    {
        no_grad guard;
        fc.parameters()[0].copy_(w);
        fc.parameters()[1].copy_(b);
    }

    Tensor x = Tensor::zeros({3, 2});
    // row 0: [10, 20]
    x.data_ptr()[0] = 10.0f;
    x.data_ptr()[1] = 20.0f;
    // row 1: [30, 40]
    x.data_ptr()[2] = 30.0f;
    x.data_ptr()[3] = 40.0f;
    // row 2: [50, 60]
    x.data_ptr()[4] = 50.0f;
    x.data_ptr()[5] = 60.0f;

    Tensor y = fc(x);

    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{3, 2}));
    // row 0: [11, 22]
    EXPECT_FLOAT_EQ(y.data_ptr()[0], 11.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[1], 22.0f);
    // row 1: [31, 42]
    EXPECT_FLOAT_EQ(y.data_ptr()[2], 31.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[3], 42.0f);
    // row 2: [51, 62]
    EXPECT_FLOAT_EQ(y.data_ptr()[4], 51.0f);
    EXPECT_FLOAT_EQ(y.data_ptr()[5], 62.0f);
}

TEST_F(LinearTest, ZeroFeaturesThrows) {
    constexpr size_t kZeroFeatures = 0;
    constexpr size_t kValidFeatures = 4;

    EXPECT_THROW((void)Linear(kZeroFeatures, kValidFeatures), std::invalid_argument);
    EXPECT_THROW((void)Linear(kValidFeatures, kZeroFeatures), std::invalid_argument);
    EXPECT_THROW((void)Linear(kZeroFeatures, kZeroFeatures), std::invalid_argument);
}
