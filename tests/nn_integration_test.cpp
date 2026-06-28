#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "grad_check.hpp"
#include "nn/linear.hpp"
#include "nn/loss.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"

using namespace helix;

class NNIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(NNIntegrationTest, EndToEndForwardBackward) {
    Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));

    Tensor x = Tensor::randn({32, 2});
    Tensor target = Tensor::randn({32, 1});

    Tensor y = net(x);
    Tensor loss = mse_loss(y, target);

    EXPECT_EQ(loss.shape().rank(), 0);  // scalar

    loss.backward();

    auto params = net.parameters();
    for (const auto& p : params) {
        EXPECT_TRUE(p.grad().shape() == p.shape());
    }
}

TEST_F(NNIntegrationTest, GradientCheck) {
    Sequential net(Linear(2, 8), ReLU(), Linear(8, 1));

    Tensor x = Tensor::randn({4, 2});  // smaller batch
    x = x + 0.1f;                      // offset to avoid relu non-differentiable point
    Tensor target = Tensor::randn({4, 1});

    auto params = net.parameters();  // size: 4

    auto func = [&target](const std::vector<Tensor>& args) {
        Tensor input = args[0];
        Tensor w1 = args[1];
        Tensor b1 = args[2];
        Tensor w2 = args[3];
        Tensor b2 = args[4];

        Tensor h = relu(input.matmul(w1) + b1);
        Tensor y = h.matmul(w2) + b2;
        return mse_loss(y, target);
    };

    std::vector<Tensor> inputs = {x};
    for (const auto& p : params) {
        inputs.push_back(p);
    }

    EXPECT_TRUE(gradient_check(func, inputs, 1e-3f, 1e-2f));
}

TEST_F(NNIntegrationTest, GraphLifecycle_MultipleBackwards) {
    Sequential net(Linear(2, 4), Linear(4, 1));
    Tensor x = Tensor::randn({32, 2});
    Tensor target = Tensor::randn({32, 1});

    Tensor y = net(x);
    Tensor loss = mse_loss(y, target);

    loss.backward();
    // HELIX currently retains graph indefinitely since it's stored in TensorImpl.
    // Calling backward again will just accumulate gradients.
    loss.backward();
    EXPECT_NO_THROW({ loss.backward(); });
}

TEST_F(NNIntegrationTest, SubgraphIsolation_Detach) {
    Linear fc1(2, 4);
    Linear fc2(4, 1);

    Tensor x = Tensor::randn({32, 2});
    Tensor h1 = fc1(x);

    Tensor h1_detached = h1.detach();
    h1_detached.set_requires_grad(true);

    Tensor y = fc2(h1_detached);
    Tensor loss = mse_loss(y, Tensor::randn({32, 1}));

    loss.backward();

    EXPECT_EQ(fc2.parameters()[0].grad().shape().vec(), (std::vector<size_t>{4, 1}));
    // Since default Tensor() has Shape{} (rank 0, numel 1), an unpopulated grad for a (2,4) weight will have rank 0.
    EXPECT_EQ(fc1.parameters()[0].grad().rank(), 0);
}
