#include <gtest/gtest.h>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
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

    // In-place modification on a leaf tensor requiring grad should throw immediately
    Tensor d = Tensor::ones(Shape({2, 2}));
    EXPECT_THROW(a.add_(d), std::runtime_error);
    EXPECT_THROW(a.zero_(), std::runtime_error);
    EXPECT_THROW(a.set_item({0, 0}, 10.0f), std::runtime_error);
    EXPECT_THROW(a.copy_(d), std::runtime_error);
}

TEST_F(AutogradTest, InplaceOperationOnNonLeafThrows) {
    Tensor a = Tensor::ones(Shape({2, 2}));
    a.set_requires_grad(true);
    Tensor b = a * 2.0f;  // b is a non-leaf tensor requiring grad

    Tensor c = Tensor::ones(Shape({2, 2}));

    // In-place on non-leaf requiring grad should throw until properly supported
    EXPECT_THROW(b.add_(c), std::runtime_error);
    EXPECT_THROW(b.zero_(), std::runtime_error);
    EXPECT_THROW(b.set_item({0, 0}, 10.0f), std::runtime_error);
    EXPECT_THROW(b.copy_(c), std::runtime_error);
}

TEST_F(AutogradTest, InplaceOperationOnNoGradSucceeds) {
    Tensor a = Tensor::ones(Shape({2, 2}));  // no grad
    Tensor b = Tensor::ones(Shape({2, 2}));  // no grad

    // In-place on no-grad should succeed silently
    EXPECT_NO_THROW(a.add_(b));
    EXPECT_EQ(a.data_ptr<float>()[0], 2.0f);
    EXPECT_NO_THROW(a.zero_());
    EXPECT_NO_THROW(a.set_item({0, 0}, 10.0f));
    EXPECT_NO_THROW(a.copy_(b));
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
