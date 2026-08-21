#include <gtest/gtest.h>

#include "autograd/engine.hpp"
#include "core/dispatcher.hpp"
#include "core/tensor.hpp"
#include "grad_check.hpp"  // Included for numerical verification
#include "optim/sgd.hpp"

using namespace helix;

class AutogradMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override { init_autograd(); }
};

// ============================================================================
// Bug 1: Missing CastBackward Node
// Principles: Zero Naive Testing (Edge Cases), Graph Lifecycle Verification (Numerical), Stress & Scale
// ============================================================================

TEST_F(AutogradMultiDTypeTest, CastBackward_Float32ToFloat64_HappyPath) {
    Tensor a(Shape({1}), DType::Float32);
    a.data_ptr<float>()[0] = 2.0f;
    a.set_requires_grad(true);

    Tensor b = Dispatcher::cast(a, DType::Float64);
    Tensor loss = b * b;  // loss = 4.0

    loss.sum().backward();

    EXPECT_TRUE(a.has_grad()) << "Graph broke at OpType::Cast! Gradient did not propagate to 'a'.";
    if (a.has_grad()) {
        EXPECT_EQ(a.grad().dtype(), DType::Float32);
        EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[0], 4.0f);
    }
}

TEST_F(AutogradMultiDTypeTest, CastBackward_Int32ToFloat32_EdgeCase) {
    Tensor a(Shape({1}), DType::Float32);
    a.data_ptr<float>()[0] = 3.0f;
    a.set_requires_grad(true);

    Tensor b(Shape({1}), DType::Int32);  // No requires_grad
    b.data_ptr<int32_t>()[0] = 2;

    Tensor loss = a + b;  // 3.0 + 2.0 = 5.0
    loss.sum().backward();

    EXPECT_TRUE(a.has_grad()) << "Graph broke during implicit Type Promotion (Cast).";
    if (a.has_grad()) {
        EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[0], 1.0f);
    }
}

TEST_F(AutogradMultiDTypeTest, CastBackward_NumericalVerification) {
    // Numerical Verification: verify that the analytical gradient from CastBackward
    // matches the numerical gradient computed via finite differences.
    auto func = [](const std::vector<Tensor>& inputs) -> Tensor {
        return Dispatcher::cast(inputs[0], DType::Float64).sum();
    };

    Tensor a(Shape({3, 3}), DType::Float32);
    float* a_ptr = a.data_ptr<float>();
    for (int i = 0; i < 9; ++i) a_ptr[i] = static_cast<float>(i) * 0.5f;
    a.set_requires_grad(true);

    // Using gradient_check with tolerance 1e-2 due to Float32 eps precision
    bool grad_correct = gradient_check(func, {a}, 1e-3f, 1e-2f);
    EXPECT_TRUE(grad_correct) << "Numerical gradient check failed for CastBackward!";
}

TEST_F(AutogradMultiDTypeTest, CastBackward_StressAndScale) {
    // Stress & Scale: High-volume iterative execution loop to catch data corruption/overflows
    Tensor a(Shape({1}), DType::Float32);
    a.data_ptr<float>()[0] = 1.0f;
    a.set_requires_grad(true);

    Tensor curr = a;
    for (int i = 0; i < 1000; ++i) {
        // Ping-pong casting to build a deep graph of 1000 Cast operations
        if (i % 2 == 0) {
            curr = Dispatcher::cast(curr, DType::Float64);
        } else {
            curr = Dispatcher::cast(curr, DType::Float32);
        }
    }

    EXPECT_NO_THROW({ curr.sum().backward(); }) << "Stress test failed: Deep Cast graph caused a crash.";

    EXPECT_TRUE(a.has_grad());
    if (a.has_grad()) {
        EXPECT_FLOAT_EQ(a.grad().data_ptr<float>()[0], 1.0f)
            << "Gradient decayed or exploded during 1000 successive casts.";
    }
}

// ============================================================================
// Bug 2: Dispatcher::clone Hardcoded Float32
// Principles: Resource Allocation & Memory Safety (Deep Copy Isolation), Memory Layout & Contiguity
// ============================================================================

TEST_F(AutogradMultiDTypeTest, DispatcherClone_Float64_MemoryIntegrity_DeepCopyIsolation) {
    Tensor a(Shape({5}), DType::Float64);
    double* a_ptr = a.data_ptr<double>();
    for (int i = 0; i < 5; ++i) {
        a_ptr[i] = i * 1.5;
    }

    Tensor cloned;
    EXPECT_NO_THROW({ cloned = Dispatcher::clone(a); }) << "clone() crashed when handling Float64 tensor.";

    ASSERT_EQ(cloned.numel(), a.numel());
    double* c_ptr = cloned.data_ptr<double>();

    // Deep Copy Isolation check: Modify original, ensure clone is unaffected
    a_ptr[0] = 999.9;
    EXPECT_NE(c_ptr[0], a_ptr[0]) << "Deep Copy Isolation failed: Clone shares memory with original tensor.";
    EXPECT_DOUBLE_EQ(c_ptr[0], 0.0) << "Memory corruption detected in clone() for Float64 at index 0";

    for (int i = 1; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(c_ptr[i], i * 1.5) << "Memory corruption detected in clone() for Float64 at index " << i;
    }
}

TEST_F(AutogradMultiDTypeTest, DispatcherClone_NonContiguous_Float64) {
    Tensor a(Shape({4, 4}), DType::Float64);
    double* a_ptr = a.data_ptr<double>();
    for (int i = 0; i < 16; ++i) a_ptr[i] = i;

    Tensor a_slice = a.slice(0, 1, 3);  // Shape {2, 4}, non-contiguous

    Tensor cloned;
    EXPECT_NO_THROW({ cloned = Dispatcher::clone(a_slice); }) << "clone() non-contiguous path crashed on Float64.";

    // Validate values
    double* c_ptr = cloned.data_ptr<double>();
    EXPECT_DOUBLE_EQ(c_ptr[0], 4.0);  // row 1, col 0
    EXPECT_DOUBLE_EQ(c_ptr[4], 8.0);  // row 2, col 0
}

// ============================================================================
// Bug 3: BackwardEngine Default Gradient Float32 Init
// Principle: Zero Naive Testing (Error Handling)
// ============================================================================

TEST_F(AutogradMultiDTypeTest, BackwardEngine_InitGrad_MatchLossDType) {
    Tensor a(Shape({1}), DType::Float64);
    a.data_ptr<double>()[0] = 2.0;
    a.set_requires_grad(true);

    Tensor loss = a * a;  // Loss is Float64
    EXPECT_EQ(loss.dtype(), DType::Float64);

    EXPECT_NO_THROW({ loss.backward(); }) << "backward() crashed during engine execution.";

    EXPECT_TRUE(a.has_grad());
    if (a.has_grad()) {
        EXPECT_EQ(a.grad().dtype(), DType::Float64) << "Gradient dtype mismatch! Engine initialized wrong DType.";
    }

    SGD optimizer({a}, 0.1f);
    EXPECT_NO_THROW({ optimizer.step(); }) << "SGD step crashed! Likely due to Parameter and Gradient dtype mismatch.";
}
