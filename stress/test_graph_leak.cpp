#include <gtest/gtest.h>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
#include "core/tensor.hpp"

using namespace helix;

class GraphLeakTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

TEST_F(GraphLeakTest, GraphDestructionAfterBackward) {
    Tensor a = Tensor::ones(Shape({1}));
    a.set_requires_grad(true);

    Tensor b = a * Tensor::full(Shape({1}), 2.0f);

    auto meta_b = static_cast<AutogradMeta*>(b.impl()->autograd_meta());
    auto grad_fn = meta_b->grad_fn();  // Strong pointer

    // Before backward, grad_fn has next_edges (points to AccumulateGrad)
    EXPECT_TRUE(grad_fn->next_edges().size() > 0) << "Graph should have edges before backward.";

    b.backward();

    // AFTER backward, PyTorch clears the graph. HELIX does NOT!
    // Let's prove HELIX does not clear it by EXPECT_TRUE.
    EXPECT_TRUE(grad_fn->next_edges().size() > 0) << "BUG PROOF: HELIX does not clear next_edges after backward!";

    // Because it doesn't clear, we can call backward again!
    b.backward();

    // And gradient accumulates!
    EXPECT_EQ(a.grad().item(), 4.0f) << "BUG PROOF: Gradient accumulated to 4.0!";
}
