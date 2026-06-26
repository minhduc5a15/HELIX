#include "nn/sequential.hpp"

#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "nn/linear.hpp"
#include "nn/relu.hpp"

using namespace helix;

class SequentialTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(SequentialTest, ShapeAndParametersTest) {
    Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));

    Tensor x = Tensor::randn({32, 2});
    Tensor y = net(x);

    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{32, 1}));

    auto params = net.parameters();
    EXPECT_EQ(params.size(), 4);  // w1, b1, w2, b2
}

TEST_F(SequentialTest, ErrorHandling_MismatchLayerShapes) {
    Sequential net(
        Linear(2, 8), Linear(4, 1)  // Expects 4, but gets 8 from previous layer
    );
    Tensor x = Tensor::randn({32, 2});
    EXPECT_THROW({ net(x); }, std::exception);
}

TEST_F(SequentialTest, EdgeCases_EmptySequential) {
    Sequential net;
    Tensor x = Tensor::randn({32, 2});
    Tensor y = net(x);
    EXPECT_EQ(y.shape().vec(), x.shape().vec());
    EXPECT_EQ(net.parameters().size(), 0);
}

TEST_F(SequentialTest, EdgeCases_LargeSequentialParametersOrder) {
    Sequential net(Linear(2, 4), ReLU(), Linear(4, 8), ReLU(), Linear(8, 16));
    auto params = net.parameters();
    EXPECT_EQ(params.size(), 6);

    EXPECT_EQ(params[0].shape().vec(), (std::vector<size_t>{2, 4}));   // w1
    EXPECT_EQ(params[1].shape().vec(), (std::vector<size_t>{4}));      // b1
    EXPECT_EQ(params[2].shape().vec(), (std::vector<size_t>{4, 8}));   // w2
    EXPECT_EQ(params[3].shape().vec(), (std::vector<size_t>{8}));      // b2
    EXPECT_EQ(params[4].shape().vec(), (std::vector<size_t>{8, 16}));  // w3
    EXPECT_EQ(params[5].shape().vec(), (std::vector<size_t>{16}));     // b3
}

TEST_F(SequentialTest, NonContiguousTensorPropagation) {
    Sequential net(Linear(2, 4), ReLU(), Linear(4, 1));
    
    // Create a contiguous tensor [2, 32] and transpose to [32, 2] -> non-contiguous
    Tensor x = Tensor::randn({2, 32});
    Tensor x_t = x.transpose(0, 1);
    EXPECT_FALSE(x_t.is_contiguous());
    EXPECT_EQ(x_t.shape().vec(), (std::vector<size_t>{32, 2}));
    
    x_t.set_requires_grad(true);

    Tensor y = net(x_t);
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{32, 1}));
    
    Tensor loss = y.sum();
    loss.backward();

    EXPECT_EQ(x_t.grad().shape().vec(), (std::vector<size_t>{32, 2}));
    EXPECT_EQ(net.parameters()[0].grad().shape().vec(), (std::vector<size_t>{2, 4}));
}
