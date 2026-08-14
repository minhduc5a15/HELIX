#include <gtest/gtest.h>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
#include "autograd/node.hpp"
#include "core/dispatcher.hpp"
#include "core/tensor.hpp"

using namespace helix;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(AutogradTest, InplaceOperationBreaksGraph) {
    Tensor a = Tensor::ones(Shape({2, 2}));
    a.set_requires_grad(true);

    Tensor b = Tensor::ones(Shape({2, 2}));
    b.set_requires_grad(true);

    // Using mul creates a SavedTensor for 'a' and 'b' in MulBackward
    Tensor c = a * b;  // c = 1

    // In-place modification on a!
    Tensor d = Tensor::ones(Shape({2, 2}));
    a.add_(d);  // a becomes 2, version increments!

    Tensor loss = c.sum();

    // Backward should throw because MulBackward tries to unpack 'a'
    EXPECT_THROW(loss.backward(), std::runtime_error);
}

TEST_F(AutogradTest, StandardTrainingLoopNoFalsePositive) {
    Tensor w = Tensor::ones(Shape({1}));
    w.set_requires_grad(true);

    Tensor x = Tensor::full(Shape({1}), 2.0f);

    // Simulate 3 epochs
    for (int epoch = 0; epoch < 3; ++epoch) {
        // Forward
        Tensor y = w * x;
        Tensor loss = y.sum();

        // Backward
        loss.backward();

        // SGD step (mutates w in-place)
        helix::Dispatcher::sgd(w, w.grad(), 0.1f);

        // No exception should be thrown above!
        // Clear grad manually as we don't have an optimizer.zero_grad() yet
        // For testing we just let it accumulate, or we can just ignore.
    }

    SUCCEED();
}

TEST_F(AutogradTest, MultipleBackwardPasses) {
    Tensor a = Tensor::ones(Shape({1}));
    a.set_requires_grad(true);
    Tensor two = Tensor::full(Shape({1}), 2.0f);
    Tensor b = a * two;

    auto meta_b = static_cast<AutogradMeta*>(b.impl()->autograd_meta());
    auto grad_fn = meta_b->grad_fn();

    // Check that graph is intact before backward
    EXPECT_TRUE(grad_fn->next_edges().size() > 0) << "Graph should have edges before backward.";

    b.backward();
    EXPECT_EQ(a.grad().item(), 2.0f);

    // AFTER backward, the graph should be cleared.
    EXPECT_EQ(grad_fn->next_edges().size(), 0) << "Graph was not cleared after backward!";

    // Calling backward again should throw because graph is destroyed!
    EXPECT_THROW(b.backward(), std::runtime_error) << "Backward should throw on cleared graph!";
}

TEST_F(AutogradTest, MultipleBackwardPasses_RetainGraph) {
    Tensor a = Tensor::ones(Shape({1}));
    a.set_requires_grad(true);
    Tensor two = Tensor::full(Shape({1}), 2.0f);
    Tensor b = a * two;

    auto meta_b = static_cast<AutogradMeta*>(b.impl()->autograd_meta());
    auto grad_fn = meta_b->grad_fn();

    // Pass retain_graph = true
    b.backward({}, true);
    EXPECT_EQ(a.grad().item(), 2.0f);

    // The graph should still be intact
    EXPECT_TRUE(grad_fn->next_edges().size() > 0) << "Graph should be retained!";

    // We can call backward again!
    b.backward();
    EXPECT_EQ(a.grad().item(), 4.0f) << "Gradient should accumulate with retain_graph!";
}
