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

// ============================================================================
// Bug 4: DType Erasure & Poisoning – More severe
// ============================================================================

TEST_F(AutogradMultiDTypeTest, AccumulateGrad_TypeMismatch_ComplexShapes) {
    // Tensor 3x4, check each element
    Tensor a(Shape({3, 4}), DType::Float32);
    float* a_ptr = a.data_ptr<float>();
    for (int i = 0; i < 12; ++i) a_ptr[i] = static_cast<float>(i) * 0.1f;
    a.set_requires_grad(true);

    // Float32 branch
    Tensor b(Shape({3, 4}), DType::Float32);
    float* b_ptr = b.data_ptr<float>();
    for (int i = 0; i < 12; ++i) b_ptr[i] = static_cast<float>(i) * 0.2f;

    // Float64 branch
    Tensor c(Shape({3, 4}), DType::Float64);
    double* c_ptr = c.data_ptr<double>();
    for (int i = 0; i < 12; ++i) c_ptr[i] = static_cast<double>(i) * 0.3;

    Tensor loss1 = a * b;          // grad = b (Float32)
    Tensor loss2 = a * c;          // grad = c (Float64)
    Tensor total = loss1 + loss2;  // promoted to Float64

    total.sum().backward({}, true);

    ASSERT_TRUE(a.has_grad());
    EXPECT_EQ(a.grad().dtype(), DType::Float32);
    float* grad = a.grad().data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
        float expected = b_ptr[i] + static_cast<float>(c_ptr[i]);
        EXPECT_FLOAT_EQ(grad[i], expected) << "Mismatch at index " << i;
    }

    a.grad().zero_();

    // Test backward multiple times with retain_graph
    Tensor total2 = loss1 + loss2;
    EXPECT_NO_THROW({
        total2.sum().backward({}, true);
        total2.sum().backward();  // accumulate again
    }) << "Multiple backward calls failed";

    grad = a.grad().data_ptr<float>();
    for (int i = 0; i < 12; ++i) {
        float expected = 2.0f * (b_ptr[i] + static_cast<float>(c_ptr[i]));
        EXPECT_FLOAT_EQ(grad[i], expected) << "Mismatch after 2 accumulations at index " << i;
    }
}

TEST_F(AutogradMultiDTypeTest, AccumulateGrad_TypeMismatch_OrderDependence) {
    // Check whether accumulation order affects results
    Tensor a(Shape({2, 2}), DType::Float32);
    float* a_ptr = a.data_ptr<float>();
    a_ptr[0] = 1.0f;
    a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f;
    a.set_requires_grad(true);

    Tensor b(Shape({2, 2}), DType::Float32);
    b.data_ptr<float>()[0] = 0.5f;
    b.data_ptr<float>()[1] = 1.5f;
    b.data_ptr<float>()[2] = 2.5f;
    b.data_ptr<float>()[3] = 3.5f;

    Tensor c(Shape({2, 2}), DType::Float64);
    c.data_ptr<double>()[0] = 0.1;
    c.data_ptr<double>()[1] = 0.2;
    c.data_ptr<double>()[2] = 0.3;
    c.data_ptr<double>()[3] = 0.4;

    Tensor loss1 = a * b;
    Tensor loss2 = a * c;

    // Order 1: loss1 first, loss2 second
    loss1.sum().backward({}, true);
    float* grad1 = a.grad().data_ptr<float>();
    // After loss1: grad = b
    for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(grad1[i], b.data_ptr<float>()[i]);

    loss2.sum().backward({}, true);
    // After loss2: grad = b + cast(c)
    for (int i = 0; i < 4; i++) {
        float expected = b.data_ptr<float>()[i] + static_cast<float>(c.data_ptr<double>()[i]);
        EXPECT_FLOAT_EQ(grad1[i], expected);
    }

    // Order 2: loss2 first, loss1 second (using a new tensor)
    Tensor a2(Shape({2, 2}), DType::Float32);
    a2.data_ptr<float>()[0] = 1.0f;
    a2.data_ptr<float>()[1] = 2.0f;
    a2.data_ptr<float>()[2] = 3.0f;
    a2.data_ptr<float>()[3] = 4.0f;
    a2.set_requires_grad(true);
    Tensor loss1b = a2 * b;
    Tensor loss2b = a2 * c;

    loss2b.sum().backward({}, true);
    float* grad2 = a2.grad().data_ptr<float>();
    // After loss2: grad = cast(c)
    for (int i = 0; i < 4; i++) EXPECT_FLOAT_EQ(grad2[i], static_cast<float>(c.data_ptr<double>()[i]));

    loss1b.sum().backward({}, true);
    // After loss1: grad = cast(c) + b
    for (int i = 0; i < 4; i++) {
        float expected = static_cast<float>(c.data_ptr<double>()[i]) + b.data_ptr<float>()[i];
        EXPECT_FLOAT_EQ(grad2[i], expected);
    }

    // Final results must be identical (regardless of order)
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(grad1[i], grad2[i]) << "Order of accumulation led to different results!";
    }
}

TEST_F(AutogradMultiDTypeTest, AccumulateGrad_WithInitialGrad) {
    // Gradient already present before backward
    Tensor a(Shape({3}), DType::Float32);
    a.data_ptr<float>()[0] = 1.0f;
    a.data_ptr<float>()[1] = 2.0f;
    a.data_ptr<float>()[2] = 3.0f;
    a.set_requires_grad(true);

    Tensor init_grad(Shape({3}), DType::Float32);
    init_grad.data_ptr<float>()[0] = 10.0f;
    init_grad.data_ptr<float>()[1] = 20.0f;
    init_grad.data_ptr<float>()[2] = 30.0f;
    (a * init_grad).sum().backward({}, true);  // Populate initial gradient

    Tensor b(Shape({3}), DType::Float64);
    b.data_ptr<double>()[0] = 0.1;
    b.data_ptr<double>()[1] = 0.2;
    b.data_ptr<double>()[2] = 0.3;
    Tensor loss = a * b;  // grad = b (Float64)
    loss.sum().backward();

    float* grad = a.grad().data_ptr<float>();
    EXPECT_EQ(a.grad().dtype(), DType::Float32);
    for (int i = 0; i < 3; i++) {
        float expected = init_grad.data_ptr<float>()[i] + static_cast<float>(b.data_ptr<double>()[i]);
        EXPECT_FLOAT_EQ(grad[i], expected) << "Mismatch with pre-existing gradient at index " << i;
    }
}

TEST_F(AutogradMultiDTypeTest, SliceBackward_DTypeInheritance_Complex) {
    // Non-contiguous slice and check specific values
    Tensor a(Shape({5, 5}), DType::Float64);
    double* a_ptr = a.data_ptr<double>();
    for (int i = 0; i < 25; ++i) a_ptr[i] = static_cast<double>(i) * 0.1;
    a.set_requires_grad(true);

    // Take slice: rows 1,2 and columns 1,2 -> creates non-contiguous
    Tensor slice1 = a.slice(0, 1, 3);       // shape (2,5)
    Tensor slice2 = slice1.slice(1, 1, 3);  // shape (2,2)
    Tensor loss = slice2 * slice2;
    loss.sum().backward();

    ASSERT_TRUE(a.has_grad());
    EXPECT_EQ(a.grad().dtype(), DType::Float64);
    double* grad = a.grad().data_ptr<double>();

    // Check each position: gradient only non-zero at (1,1),(1,2),(2,1),(2,2)
    // Value = 2 * original value
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            double expected = 0.0;
            if (i >= 1 && i <= 2 && j >= 1 && j <= 2) {
                double val = static_cast<double>(i * 5 + j) * 0.1;
                expected = 2.0 * val;
            }
            EXPECT_DOUBLE_EQ(grad[i * 5 + j], expected) << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(AutogradMultiDTypeTest, SliceBackward_WithExistingGrad) {
    // Slice with pre-existing gradient (test accumulation)
    Tensor a(Shape({4, 4}), DType::Float64);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) a.data_ptr<double>()[i * 4 + j] = static_cast<double>(i * 4 + j);
    a.set_requires_grad(true);

    Tensor init_grad(Shape({4, 4}), DType::Float64);
    for (int i = 0; i < 16; i++) init_grad.data_ptr<double>()[i] = static_cast<double>(i) * 0.5;
    (a * init_grad).sum().backward({}, true);  // Populate initial gradient

    Tensor slice = a.slice(0, 1, 3);  // rows 1,2
    Tensor loss = slice * slice;
    loss.sum().backward();  // add gradient

    double* grad = a.grad().data_ptr<double>();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = init_grad.data_ptr<double>()[i * 4 + j];
            if (i >= 1 && i <= 2) {
                expected += 2.0 * a.data_ptr<double>()[i * 4 + j];
            }
            EXPECT_DOUBLE_EQ(grad[i * 4 + j], expected)
                << "Mismatch with existing gradient at (" << i << "," << j << ")";
        }
    }
}

TEST_F(AutogradMultiDTypeTest, Optimizer_ZeroGrad_DTypeAndValue) {
    // zero_grad must reset to 0 and keep dtype unchanged
    Tensor a(Shape({5}), DType::Float64);
    for (int i = 0; i < 5; i++) a.data_ptr<double>()[i] = (double)(i + 1);
    a.set_requires_grad(true);
    Tensor loss = a * a;
    loss.sum().backward();

    ASSERT_TRUE(a.has_grad());
    EXPECT_EQ(a.grad().dtype(), DType::Float64);
    double* grad = a.grad().data_ptr<double>();
    for (int i = 0; i < 5; i++) EXPECT_NE(grad[i], 0.0);  // not zero

    SGD opt({a}, 0.1);
    opt.zero_grad();

    EXPECT_EQ(a.grad().dtype(), DType::Float64);
    grad = a.grad().data_ptr<double>();
    for (int i = 0; i < 5; i++) EXPECT_DOUBLE_EQ(grad[i], 0.0) << "zero_grad did not reset to 0 at index " << i;

    // Check with multiple parameters, different dtypes
    Tensor b(Shape({5}), DType::Float32);
    for (int i = 0; i < 5; i++) b.data_ptr<float>()[i] = (float)(i + 1);
    b.set_requires_grad(true);
    // Create loss to get gradient for b
    std::cout << "a shape: " << a.shape().to_string() << "\n";
    std::cout << "b shape: " << b.shape().to_string() << "\n";
    Tensor loss2 = a * Dispatcher::cast(b, DType::Float64);
    loss2.sum().backward();

    ASSERT_TRUE(b.has_grad());
    EXPECT_EQ(b.grad().dtype(), DType::Float32);

    SGD opt2({a, b}, 0.1);
    opt2.zero_grad();

    EXPECT_EQ(a.grad().dtype(), DType::Float64);
    EXPECT_EQ(b.grad().dtype(), DType::Float32);
    double* ag = a.grad().data_ptr<double>();
    float* bg = b.grad().data_ptr<float>();
    for (int i = 0; i < 5; i++) EXPECT_DOUBLE_EQ(ag[i], 0.0);
    for (int i = 0; i < 5; i++) EXPECT_FLOAT_EQ(bg[i], 0.0f);
}

TEST_F(AutogradMultiDTypeTest, Optimizer_ZeroGrad_MultipleSteps) {
    // Repeat many steps, ensure zero_grad always works
    Tensor a(Shape({2}), DType::Float64);
    a.data_ptr<double>()[0] = 1.0;
    a.data_ptr<double>()[1] = 2.0;
    a.set_requires_grad(true);
    SGD opt({a}, 0.1);

    for (int step = 0; step < 5; ++step) {
        Tensor loss = a * a;
        loss.sum().backward();

        ASSERT_TRUE(a.has_grad());
        double* grad = a.grad().data_ptr<double>();
        EXPECT_NE(grad[0], 0.0);
        EXPECT_NE(grad[1], 0.0);

        opt.zero_grad();
        EXPECT_EQ(a.grad().dtype(), DType::Float64);
        grad = a.grad().data_ptr<double>();
        EXPECT_DOUBLE_EQ(grad[0], 0.0);
        EXPECT_DOUBLE_EQ(grad[1], 0.0);
    }
}