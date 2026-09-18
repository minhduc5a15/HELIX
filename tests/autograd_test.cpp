#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
#include "core/dispatcher.hpp"
#include "core/tensor.hpp"

using namespace helix;

class AutogradTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the Autograd Engine Hooks
        init_autograd();
    }
};

TEST(AutogradInitializationTest, ThreadSeesBuilderInitializedAfterFirstAccess) {
    std::atomic<bool> first_access_done{false};
    std::atomic<bool> initialized{false};
    bool has_grad = false;

    std::thread worker([&] {
        {
            no_grad guard;
            EXPECT_EQ(Dispatcher::get_graph_builder(), nullptr);
        }
        first_access_done.store(true);
        while (!initialized.load()) std::this_thread::yield();

        Tensor a({1.0f}, Shape{1});
        a.set_requires_grad(true);
        has_grad = (a * 2.0f).requires_grad();
    });

    while (!first_access_done.load()) std::this_thread::yield();
    init_autograd();
    initialized.store(true);
    worker.join();
    EXPECT_TRUE(has_grad);
}

TEST_F(AutogradTest, SimpleAddMul) {
    Tensor a({2.0f}, Shape{1});
    Tensor b({3.0f}, Shape{1});

    a.set_requires_grad(true);
    b.set_requires_grad(true);

    // c = a + b * a -> c = 2 + 3 * 2 = 8
    Tensor c = a + b * a;

    EXPECT_FLOAT_EQ(c.item(), 8.0f);

    c.backward();

    // dc/da = 1 + b = 4
    // dc/db = a = 2
    EXPECT_FLOAT_EQ(a.grad().item(), 4.0f);
    EXPECT_FLOAT_EQ(b.grad().item(), 2.0f);
}

TEST_F(AutogradTest, MeanGradient) {
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    a.set_requires_grad(true);

    Tensor c = a.mean();  // c = 2.5

    EXPECT_FLOAT_EQ(c.item(), 2.5f);

    c.backward();

    // dc/da = 1/4 = 0.25 for all elements
    const float* grad_ptr = a.grad().data_ptr();
    EXPECT_FLOAT_EQ(grad_ptr[0], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[1], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[2], 0.25f);
    EXPECT_FLOAT_EQ(grad_ptr[3], 0.25f);
}

TEST_F(AutogradTest, Detach) {
    Tensor a({2.0f}, Shape{1});
    a.set_requires_grad(true);

    Tensor b = a.detach();
    EXPECT_FALSE(b.requires_grad());

    Tensor c = a * b;
    c.backward();

    // c = a * 2.0 (since b is detached, b is constant 2.0)
    // dc/da = b = 2.0
    EXPECT_FLOAT_EQ(a.grad().item(), 2.0f);
}

TEST_F(AutogradTest, ErrorHandling_NonScalarWithoutGradOutputs) {
    // Calling backward() on a non-scalar output without grad_outputs should throw std::runtime_error
    Tensor a({1.0f, 2.0f}, Shape{2});
    a.set_requires_grad(true);

    Tensor scale({2.0f, 2.0f}, Shape{2});
    Tensor b = a * scale;
    EXPECT_THROW({ b.backward(); }, std::runtime_error);
}

TEST_F(AutogradTest, ErrorHandling_NoRequiresGrad) {
    // Calling backward() on a tensor that does not require grad should throw std::runtime_error
    Tensor a({1.0f}, Shape{1});
    // requires_grad is false by default
    EXPECT_THROW({ a.backward(); }, std::runtime_error);
}

TEST_F(AutogradTest, NonScalarWithGradOutputs) {
    // Calling backward(grad_outputs) on a non-scalar tensor should work and use the provided gradients
    Tensor x({1.0f, 2.0f}, Shape{2});
    x.set_requires_grad(true);

    Tensor scale({2.0f, 2.0f}, Shape{2});
    Tensor y = x * scale;

    // grad_outputs matching shape of y
    Tensor grad_out({1.0f, 3.0f}, Shape{2});
    y.backward({grad_out});

    // dy/dx = scale = 2.0, so x.grad = grad_out * scale = [2.0, 6.0]
    EXPECT_FLOAT_EQ(x.grad().data_ptr()[0], 2.0f);
    EXPECT_FLOAT_EQ(x.grad().data_ptr()[1], 6.0f);
}

// UAF Proof of Concept
using namespace helix;

Tensor build_graph_and_destroy_leaf() {
    Tensor w(Shape{2, 2});
    w.set_requires_grad(true);

    Tensor x(Shape{2, 2});

    Tensor y = w * x;
    return y;
}

TEST_F(AutogradTest, AccumulateGrad_UseAfterFree) {
    Tensor y = build_graph_and_destroy_leaf();
    y.sum().backward();
}

TEST_F(AutogradTest, AccumulateGrad_BroadcastedViewSafety) {
    Tensor x({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2});
    x.set_requires_grad(true);

    Tensor y = x.sum();
    y.backward();

    // The gradient should have been cloned by AccumulateGrad if it was a broadcasted view,
    // so in-place operations like zero_() should succeed without throwing overlapping memory errors.
    EXPECT_NO_THROW({ x.grad().zero_(); });
}

TEST_F(AutogradTest, BackwardTwiceWithoutRetainGraphThrows) {
    Tensor a({1.0f, 2.0f}, {2});
    a.set_requires_grad(true);
    Tensor b({3.0f, 4.0f}, {2});
    b.set_requires_grad(true);

    Tensor c = a * b;
    Tensor d = c.sum();

    // First backward should succeed
    EXPECT_NO_THROW(d.backward());
    EXPECT_EQ(a.grad().data_ptr()[0], 3.0f);

    // Second backward should throw because graph was destroyed
    EXPECT_THROW(
        {
            try {
                d.backward();
            } catch (const std::runtime_error& e) {
                // Check if error message contains the expected string
                EXPECT_STREQ(
                    "RuntimeError: Trying to backward through the graph a second time. Specify retain_graph=true if "
                    "you need to backward through the graph a second time.",
                    e.what()
                );
                throw;
            }
        },
        std::runtime_error
    );
}

TEST_F(AutogradTest, IntegerTensorsCannotRequireGrad) {
    Tensor a(Shape({2}), DType::Int32);
    EXPECT_THROW(a.set_requires_grad(true), std::runtime_error);

    Tensor b(Shape({2}), DType::Int64);
    EXPECT_THROW(b.set_requires_grad(true), std::runtime_error);

    // Float should not throw
    Tensor c(Shape({2}), DType::Float32);
    EXPECT_NO_THROW(c.set_requires_grad(true));
}

TEST_F(AutogradTest, BackwardOnIntegerOutputThrows) {
    Tensor a({1.0f, 2.0f}, Shape({2}));
    a.set_requires_grad(true);

    Tensor b = a * 2.0f;
    // Cast to Int32
    Tensor c(b.shape(), DType::Int32);
    // Even if we try to backward manually on an integer output, the framework should either ignore or throw.
    // In our design, casting breaks the graph since integers don't have grad.
    // Wait, let's see if backward on C throws. c does not require grad.
    EXPECT_THROW(c.backward(), std::runtime_error);
}

TEST_F(AutogradTest, NoGrad_BasicScope) {
    Tensor a({2.0f}, Shape{1});
    Tensor b({3.0f}, Shape{1});
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    Tensor c;
    {
        no_grad guard;
        c = a + b * a;
        EXPECT_FALSE(c.requires_grad());
        EXPECT_EQ(c.impl()->autograd_meta(), nullptr);
    }

    // Outside no_grad, autograd should track operations again
    Tensor d = a + b;
    EXPECT_TRUE(d.requires_grad());
    EXPECT_NE(d.impl()->autograd_meta(), nullptr);
    EXPECT_NE(d.impl()->autograd_meta()->grad_fn(), nullptr);
}

TEST_F(AutogradTest, NoGrad_NestedScope) {
    Tensor a({2.0f}, Shape{1});
    a.set_requires_grad(true);

    {
        no_grad outer;
        Tensor b = a * 2.0f;
        EXPECT_FALSE(b.requires_grad());

        {
            no_grad inner;
            Tensor c = a * 3.0f;
            EXPECT_FALSE(c.requires_grad());
        }

        // Still in outer scope: must remain disabled
        Tensor d = a * 4.0f;
        EXPECT_FALSE(d.requires_grad());
    }

    // Now outside both: must be re-enabled
    Tensor e = a * 5.0f;
    EXPECT_TRUE(e.requires_grad());
}

TEST_F(AutogradTest, NoGrad_ExceptionSafety) {
    Tensor a({2.0f}, Shape{1});
    a.set_requires_grad(true);

    try {
        no_grad guard;
        Tensor b = a * 2.0f;
        EXPECT_FALSE(b.requires_grad());
        throw std::runtime_error("simulated exception");
    } catch (const std::runtime_error&) {
        // Exception caught, verify guard dtor properly restored builder
    }

    Tensor c = a * 3.0f;
    EXPECT_TRUE(c.requires_grad());
}

TEST_F(AutogradTest, NoGrad_MultiThreadIsolation) {
    Tensor a({2.0f}, Shape{1});
    a.set_requires_grad(true);

    std::atomic<bool> thread2_ready{false};
    std::atomic<bool> thread2_done{false};
    bool thread2_has_grad = false;

    // Spawn thread 2 which computes with autograd while thread 1 is in no_grad
    std::thread t2([&]() {
        Tensor x({3.0f}, Shape{1});
        x.set_requires_grad(true);
        while (!thread2_ready.load()) {
            std::this_thread::yield();
        }
        Tensor y = x * 2.0f;
        thread2_has_grad = y.requires_grad();
        thread2_done.store(true);
    });

    {
        no_grad guard;
        thread2_ready.store(true);
        while (!thread2_done.load()) {
            std::this_thread::yield();
        }
        // Thread 1 must have autograd disabled
        Tensor b = a * 2.0f;
        EXPECT_FALSE(b.requires_grad());
    }

    t2.join();
    // Thread 2 must NOT have been disabled by Thread 1's no_grad
    EXPECT_TRUE(thread2_has_grad);
}
