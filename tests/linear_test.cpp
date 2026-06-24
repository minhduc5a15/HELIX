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
    Tensor x = Tensor::randn({2, 4});
    Tensor w = Tensor::randn({4, 8});
    Tensor b = Tensor::randn({8});

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

TEST_F(LinearTest, StressTest) {
    Linear fc(4, 8);
    for (int i = 0; i < 100; ++i) {
        Tensor x = Tensor::randn({32, 4});
        Tensor y = fc(x);
        Tensor loss = y.sum();
        loss.backward();
    }
    SUCCEED();
}
