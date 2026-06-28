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

TEST_F(SequentialTest, SequentialForward) {
    // Verify Sequential forward pass with known weights
    // Net: Linear(2, 2) -> ReLU() -> Linear(2, 1)
    Sequential net(Linear(2, 2), ReLU(), Linear(2, 1));
    
    // Linear 1
    Tensor w1 = Tensor::zeros({2, 2});
    w1.data_ptr()[0] = 1.0f;  w1.data_ptr()[1] = -1.0f;
    w1.data_ptr()[2] = 2.0f;  w1.data_ptr()[3] = -2.0f;
    
    Tensor b1 = Tensor::zeros({2});
    b1.data_ptr()[0] = 0.5f;
    b1.data_ptr()[1] = 1.0f;
    
    net.parameters()[0].copy_(w1);
    net.parameters()[1].copy_(b1);
    
    // Linear 2
    Tensor w2 = Tensor::zeros({2, 1});
    w2.data_ptr()[0] = 2.0f;
    w2.data_ptr()[1] = 3.0f;
    
    Tensor b2 = Tensor::zeros({1});
    b2.data_ptr()[0] = -0.5f;
    
    net.parameters()[2].copy_(w2);
    net.parameters()[3].copy_(b2);
    
    // Input [1, 2] values: [1.0, 2.0]
    Tensor x = Tensor::zeros({1, 2});
    x.data_ptr()[0] = 1.0f;
    x.data_ptr()[1] = 2.0f;
    
    // Linear 1 Output: x * w1 + b1 = [1*1 + 2*2 + 0.5, 1*(-1) + 2*(-2) + 1.0] = [5.5, -4.0]
    // ReLU Output: max(Linear 1 Output, 0) = [5.5, 0.0]
    // Linear 2 Output: relu * w2 + b2 = [5.5*2 + 0*3 - 0.5] = [10.5]
    
    Tensor y = net(x);
    
    EXPECT_EQ(y.shape().vec(), (std::vector<size_t>{1, 1}));
    EXPECT_FLOAT_EQ(y.data_ptr()[0], 10.5f);
}

TEST_F(SequentialTest, SequentialParameters) {
    // Explicitly check parameters of Sequential are correct
    Sequential net(Linear(2, 4), ReLU(), Linear(4, 1));
    auto params = net.parameters();
    
    EXPECT_EQ(params.size(), 4);
    EXPECT_EQ(params[0].shape().vec(), (std::vector<size_t>{2, 4})); // w1
    EXPECT_EQ(params[1].shape().vec(), (std::vector<size_t>{4}));    // b1
    EXPECT_EQ(params[2].shape().vec(), (std::vector<size_t>{4, 1})); // w2
    EXPECT_EQ(params[3].shape().vec(), (std::vector<size_t>{1}));    // b2
}

