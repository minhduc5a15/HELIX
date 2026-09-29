#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
#include "autograd/node.hpp"
#include "backend/cpu_backend.hpp"
#include "core/allocator.hpp"
#include "core/autotuner.hpp"
#include "core/dispatcher.hpp"
#include "core/nd_iterator.hpp"
#include "core/tensor.hpp"
#include "core/tensor_factory.hpp"
#include "nn/loss.hpp"

namespace helix {
    void avx2_dot_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void avx2_micro_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void openmp_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
}  // namespace helix

using namespace helix;

extern std::atomic<size_t> g_total_allocated;

// 1. Issue 1: Allocator underflow check
TEST(AuditRemediationTest, AllocatorUnderflowPrevention) {
    MemoryPool& pool = MemoryPool::get_instance();
    pool.reset();

    const size_t initial = g_total_allocated.load();
    void* p1 = pool.allocate(256);
    EXPECT_GT(g_total_allocated.load(), initial);

    pool.deallocate(p1, 256);

    // Cache hit on re-allocation
    void* p2 = pool.allocate(256);
    EXPECT_GT(g_total_allocated.load(), initial);

    pool.deallocate(p2, 256);
    EXPECT_EQ(g_total_allocated.load(), initial);
}

// 2. Issue 7: Thread cache epoch invalidation on exit
TEST(AuditRemediationTest, AllocatorThreadExitPostResetEpoch) {
    MemoryPool& pool = MemoryPool::get_instance();
    pool.reset();

    std::atomic<bool> thread_allocated{false};
    std::atomic<bool> reset_done{false};

    std::thread worker([&]() {
        void* p = pool.allocate(512);
        pool.deallocate(p, 512);
        thread_allocated.store(true);

        while (!reset_done.load()) {
            std::this_thread::yield();
        }
        // Thread exits while holding cached block from older epoch
    });

    while (!thread_allocated.load()) {
        std::this_thread::yield();
    }

    pool.reset();  // Increments epoch and clears global bins
    reset_done.store(true);

    worker.join();

    // After thread exit, stale epoch blocks must NOT be in global pool
    EXPECT_EQ(pool.get_global_pool_size(512), 0);
}

// 3. Issue 2: AVX2 dot matmul mathematical equivalence
TEST(AuditRemediationTest, AVX2DotMatmulEquivalence) {
    constexpr size_t M = 8, K = 8, N = 8;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 0.0f);
    for (size_t i = 0; i < 8; ++i) {
        B[i * N + i] = 1.0f;  // Identity
    }

    std::vector<float> out_dot(M * N, 0.0f);
    avx2_dot_matmul(A.data(), B.data(), out_dot.data(), M, K, N);

    std::vector<float> out_micro(M * N, 0.0f);
    avx2_micro_matmul(A.data(), B.data(), out_micro.data(), M, K, N);

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(out_dot[i], 1.0f, 1e-5f);
        EXPECT_NEAR(out_dot[i], out_micro[i], 1e-5f);
    }
}

// 4. Issue 3: NDIterator and BinaryNDIterator zero dimension safety
TEST(AuditRemediationTest, NDIteratorZeroDimensionSafety) {
    Shape zero_shape{2, 0};
    EXPECT_NO_THROW({
        NDIterator it(zero_shape);
        it.init_from_flat(0);
    });

    EXPECT_NO_THROW({
        BinaryNDIterator bit(zero_shape);
        bit.init_from_flat(0);
    });

    // Also test Tensor::copy_ with zero-size
    Tensor t1 = Tensor::empty(Shape{2, 0});
    Tensor t2 = Tensor::empty(Shape{2, 0});
    EXPECT_NO_THROW(t1.copy_(t2));
}

// 5. Issue 4: Negative stride in-place 2D addition
TEST(AuditRemediationTest, NegativeStrideInplace2D) {
    // 2x3 tensor
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    Tensor a(data, Shape{2, 3});
    Tensor b(data, Shape{2, 3});

    // In-place addition with strided views
    Tensor a_sub = a.slice(1, 0, 2);  // 2x2 view
    Tensor b_sub = b.slice(1, 0, 2);
    EXPECT_NO_THROW(a_sub.add_(b_sub));

    EXPECT_FLOAT_EQ(a.item({0, 0}), 2.0f);
    EXPECT_FLOAT_EQ(a.item({0, 1}), 4.0f);
}

// 6. Issue 5: AutoTuner concurrent access safety
TEST(AuditRemediationTest, AutoTunerConcurrentAccess) {
    AutoTuner& tuner = AutoTuner::get_instance();
    tuner.reset_for_testing();

    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            size_t threshold = tuner.get_omp_threshold();
            EXPECT_GT(threshold, 0);
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }
}

// 7. Issue 6: TensorFactory::randn multi-threaded safety
TEST(AuditRemediationTest, RandnMultithreadSafety) {
    std::atomic<bool> start_flag{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int k = 0; k < 50; ++k) {
                Tensor t = TensorFactory::randn(Shape{4, 4});
                EXPECT_EQ(t.numel(), 16);
            }
        });
    }

    start_flag.store(true, std::memory_order_release);
    for (auto& t : threads) {
        t.join();
    }
}

// 8. Issue 10: Tensor item and set_item bounds check
TEST(AuditRemediationTest, TensorItemBoundsCheck) {
    Tensor t(std::vector<float>{1, 2, 3, 4}, Shape{2, 2});

    // Valid indices
    EXPECT_NO_THROW((void)t.item({0, 0}));
    EXPECT_NO_THROW((void)t.item({1, 1}));

    // Rank mismatch
    EXPECT_THROW((void)t.item({0}), std::invalid_argument);
    EXPECT_THROW((void)t.item({0, 0, 0}), std::invalid_argument);

    // Dimension out-of-range
    EXPECT_THROW((void)t.item({2, 0}), std::out_of_range);
    EXPECT_THROW((void)t.item({0, 2}), std::out_of_range);
    EXPECT_THROW((void)t.item({100, 100}), std::out_of_range);

    // set_item out-of-range
    EXPECT_THROW(t.set_item({2, 0}, 10.0f), std::out_of_range);
    EXPECT_THROW(t.set_item({0, 2}, 10.0f), std::out_of_range);
}

// 9. Issue 11: CrossEntropy zero dimensions validation
TEST(AuditRemediationTest, CrossEntropyZeroDimensionsThrows) {
    Tensor empty_batch(Shape{0, 5});
    Tensor empty_target(Shape{0, 5});
    EXPECT_THROW(Dispatcher::cross_entropy(empty_batch, empty_target), std::invalid_argument);

    Tensor empty_classes(Shape{5, 0});
    Tensor target_classes(Shape{5, 0});
    EXPECT_THROW(Dispatcher::cross_entropy(empty_classes, target_classes), std::invalid_argument);
}

// 10. Issue 8: Autograd cycle detection in computation graph
class CyclicDummyNode : public Node {
public:
    std::vector<Tensor> backward(const std::vector<Tensor>& grad_outputs) override { return grad_outputs; }
};

TEST(AuditRemediationTest, AutogradCycleDetectionThrows) {
    init_autograd();

    auto nodeA = std::make_shared<CyclicDummyNode>();
    auto nodeB = std::make_shared<CyclicDummyNode>();

    // Create a cycle: nodeA -> nodeB -> nodeA
    nodeA->add_next_edge(nodeB);
    nodeB->add_next_edge(nodeA);

    Tensor root_tensor({1.0f}, Shape{1});
    root_tensor.set_requires_grad(true);
    root_tensor.impl()->autograd_meta()->set_grad_fn(nodeA);

    // Running backward engine on cyclic graph must detect cycle and throw
    BackwardEngine engine;
    EXPECT_THROW(engine.run(root_tensor), std::runtime_error);
}

// -----------------------------------------------------------------------------
// Suite 2: Technical Audit Report Remediations (10 New Issues)
// -----------------------------------------------------------------------------

// 1. Issue 01: Forward-Overlapping Self-Copy Aliasing
TEST(AuditRemediationTest2, CopyForwardOverlapSelfAliasing) {
    Tensor t(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5});
    Tensor dst = t.slice(0, 1, 5);  // points to t[1..4]
    Tensor src = t.slice(0, 0, 4);  // points to t[0..3]
    dst.copy_(src);

    EXPECT_FLOAT_EQ(t.item({0}), 1.0f);
    EXPECT_FLOAT_EQ(t.item({1}), 1.0f);
    EXPECT_FLOAT_EQ(t.item({2}), 2.0f);
    EXPECT_FLOAT_EQ(t.item({3}), 3.0f);
    EXPECT_FLOAT_EQ(t.item({4}), 4.0f);
}

// 2. Issue 02: Zero Rank-2 Non-Contiguous DTypes
TEST(AuditRemediationTest2, ZeroRank2NonContiguousDTypes) {
    // Int32
    Tensor t_i32 = Tensor::ones(Shape{3, 3}, DType::Int32);
    Tensor sub_i32 = t_i32.slice(1, 0, 2);
    EXPECT_NO_THROW(sub_i32.zero_());
    EXPECT_FLOAT_EQ(sub_i32.item({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(sub_i32.item({1, 1}), 0.0f);

    // Int64
    Tensor t_i64 = Tensor::ones(Shape{3, 3}, DType::Int64);
    Tensor sub_i64 = t_i64.slice(1, 0, 2);
    EXPECT_NO_THROW(sub_i64.zero_());
    EXPECT_FLOAT_EQ(sub_i64.item({0, 0}), 0.0f);

    // Float64
    Tensor t_f64 = Tensor::ones(Shape{3, 3}, DType::Float64);
    Tensor sub_f64 = t_f64.slice(1, 0, 2);
    EXPECT_NO_THROW(sub_f64.zero_());
    EXPECT_FLOAT_EQ(sub_f64.item({0, 0}), 0.0f);
}

// 3. Issue 03: Rank > 8 Throws Clean Exception Before OpenMP
TEST(AuditRemediationTest2, RankGreaterThan8ThrowsBeforeOpenMP) {
    Shape rank9_shape{2, 2, 2, 2, 2, 2, 2, 2, 2};
    Tensor t = Tensor::ones(rank9_shape).transpose(0, 1);
    EXPECT_THROW((void)t.clone(), std::invalid_argument);
    EXPECT_THROW(t.zero_(), std::invalid_argument);

    Tensor other = Tensor::ones(rank9_shape).transpose(0, 1);
    EXPECT_THROW(t.add_(other), std::invalid_argument);
}

// 4. Issue 04: ComputeOffsetFromFlat Zero Shape Safety
TEST(AuditRemediationTest2, ComputeOffsetFromFlatZeroShapeSafety) {
    Shape zero_shape{2, 0};
    Stride zero_stride = Stride::compute_contiguous(zero_shape);
    ptrdiff_t off = BinaryNDIterator::compute_offset_from_flat(0, zero_shape, zero_stride);
    EXPECT_EQ(off, 0);
}

// 5. Issue 05: Zero OpenMP Work Sharing Disjoint
TEST(AuditRemediationTest2, ZeroOpenMPWorkSharingDisjoint) {
    Tensor t = Tensor::ones(Shape{4, 4, 4});
    Tensor sub = t.slice(0, 1, 3).slice(1, 1, 3);
    sub.zero_();

    for (size_t i = 0; i < sub.shape()[0]; ++i) {
        for (size_t j = 0; j < sub.shape()[1]; ++j) {
            for (size_t k = 0; k < sub.shape()[2]; ++k) {
                EXPECT_FLOAT_EQ(sub.item({i, j, k}), 0.0f);
            }
        }
    }
}

// 6. Issue 06: TensorImpl Allocation Size Wraparound Throws
TEST(AuditRemediationTest2, TensorImplAllocationWraparoundThrows) {
    size_t huge_dim = (std::numeric_limits<size_t>::max() / 8) + 10;
    EXPECT_THROW((void)Tensor(Shape{huge_dim}, DType::Float64), std::overflow_error);
}

// 7. Issue 07: CrossEntropy Differentiable Target Throws
TEST(AuditRemediationTest2, CrossEntropyDifferentiableTargetThrows) {
    init_autograd();
    Tensor pred = Tensor::randn(Shape{2, 3});
    pred.set_requires_grad(true);
    Tensor target = Tensor::zeros(Shape{2, 3});
    target.set_requires_grad(true);

    EXPECT_THROW((void)cross_entropy_loss(pred, target), std::invalid_argument);
}

// 8. Issue 08: Negative Stride Explicit Signed Cast
TEST(AuditRemediationTest2, NegativeStrideExplicitSignedCast) {
    Stride st({10, -5});
    EXPECT_EQ(st.compute_offset({1, 2}), 0);
    EXPECT_EQ(st.compute_offset({2, 5}), -5);
}

// 9. Issue 09: MemoryPool Worker Spawn No Mutex Contention
TEST(AuditRemediationTest2, MemoryPoolWorkerSpawnNoMutexContention) {
    constexpr size_t NUM_THREADS = 8;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([]() {
            MemoryPool& pool = MemoryPool::get_instance();
            for (size_t k = 0; k < 100; ++k) {
                void* p = pool.allocate(128);
                pool.deallocate(p, 128);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

// 10. Issue 10: OpenMP MatMul K=0 Fallback Zeroed
TEST(AuditRemediationTest2, OpenMPMatmulKZeroFallbackZeroed) {
    constexpr size_t M = 128, K = 0, N = 128;
    std::vector<float> A(1, 0.0f);
    std::vector<float> B(1, 0.0f);
    std::vector<float> C(M * N, 999.0f);

    openmp_matmul(A.data(), B.data(), C.data(), M, K, N);

    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], 0.0f);
    }
}

// 11. Issue 11: Inplace Operations With Grad RHS Throws
TEST(AuditRemediationTest2, InplaceOperationsWithGradRHSThrows) {
    init_autograd();
    Tensor a = Tensor::zeros(Shape{2, 2});  // requires_grad = false
    Tensor b = Tensor::ones(Shape{2, 2});
    b.set_requires_grad(true);

    // a.add_(b) must throw because b requires grad and in-place tracking is unsupported
    EXPECT_THROW(a.add_(b), std::runtime_error);

    // dst.copy_(src) where src requires grad must also throw
    Tensor dst = Tensor::zeros(Shape{2, 2});
    EXPECT_THROW(dst.copy_(b), std::runtime_error);
}

// 12. Issue 12: CPUBackend CrossEntropy Zero Dimensions Safety
TEST(AuditRemediationTest2, CPUBackendCrossEntropyZeroDimensionsSafety) {
    float loss_out = 999.0f;
    float pred[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float target[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    float log_softmax_out[4] = {0.0f};

    // Test N = 0 (prevents division by zero NaN)
    CPUBackend::cross_entropy<float>(pred, target, &loss_out, log_softmax_out, 0, 4);
    EXPECT_FLOAT_EQ(loss_out, 0.0f);
    EXPECT_FALSE(std::isnan(loss_out));

    // Test C = 0 (prevents out-of-bounds heap read)
    loss_out = 999.0f;
    CPUBackend::cross_entropy<float>(pred, target, &loss_out, log_softmax_out, 2, 0);
    EXPECT_FLOAT_EQ(loss_out, 0.0f);
    EXPECT_FALSE(std::isnan(loss_out));
}

// 13. Issue 13: Backward Exception Clears Graph Edges (RAII Scope Guard)
TEST(AuditRemediationTest2, BackwardExceptionClearsGraphEdgesRAII) {
    init_autograd();
    Tensor a = Tensor::ones(Shape{2, 2});
    a.set_requires_grad(true);
    Tensor b = Tensor::ones(Shape{2, 2});
    b.set_requires_grad(true);
    Tensor c = a * b;
    Tensor loss = c.sum();

    // Mutate a's storage version so MulBackward's saved_a_.unpack() throws
    a.increment_version();

    auto meta_loss = static_cast<AutogradMeta*>(loss.impl()->autograd_meta());
    auto grad_fn = meta_loss->grad_fn();
    ASSERT_NE(grad_fn, nullptr);
    EXPECT_FALSE(grad_fn->next_edges().empty());

    // backward() must throw due to in-place mutation of saved tensor
    EXPECT_THROW(loss.backward(), std::runtime_error);

    // After exception unwinding, RAII GraphCleaner must have cleared next_edges_
    EXPECT_TRUE(grad_fn->next_edges().empty());
}

// 14. Issue 04: MemoryPool::deallocate integer overflow safety
TEST(AuditRemediationTest2, MemoryPoolDeallocateIntegerOverflowSafety) {
    MemoryPool& pool = MemoryPool::get_instance();
    const size_t initial_allocated = g_total_allocated.load();

    // Allocate a legitimate block
    void* ptr = pool.allocate(128);
    const size_t allocated = g_total_allocated.load();
    EXPECT_GT(allocated, initial_allocated);

    // Call deallocate with an extreme size that would overflow (bytes + 31)
    const size_t huge_bytes = std::numeric_limits<size_t>::max() - 10;
    // With defensive validation, this call must early-return without altering tracking or corrupting bins
    pool.deallocate(ptr, huge_bytes);
    EXPECT_EQ(g_total_allocated.load(), allocated);

    // Clean up legitimate block
    pool.deallocate(ptr, 128);
    EXPECT_EQ(g_total_allocated.load(), initial_allocated);
}

// 15. Issue 05: Stride::compute_contiguous signed overflow protection
TEST(AuditRemediationTest2, StrideComputeContiguousSignedOverflowThrows) {
    // A shape whose dimension product exceeds PTRDIFF_MAX (e.g. 1ULL << 63)
    // must throw std::overflow_error instead of silently wrapping into negative strides
    const size_t huge_dim = 1ULL << 63;
    Shape huge_shape({2, huge_dim});
    EXPECT_THROW(Stride::compute_contiguous(huge_shape), std::overflow_error);
}

// 16. Issue 06: Tensor::version and increment_version null storage safety
TEST(AuditRemediationTest2, TensorVersionNullStorageSafety) {
    // Construct unbacked TensorImpl with null storage
    auto impl =
        std::make_shared<TensorImpl>(nullptr, 0, Shape{2, 2}, Stride({2, 1}), DType::Float32, Device(DeviceType::CPU));
    Tensor t(impl);

    // Must return 0 without SIGSEGV
    EXPECT_EQ(t.version(), 0u);

    // Must safely no-op without SIGSEGV
    EXPECT_NO_THROW(t.increment_version());
    EXPECT_EQ(t.version(), 0u);
}

// 17. Issue 07: TensorFactory::randn produces independent sequences across distinct threads
TEST(AuditRemediationTest2, TensorRandnIndependentAcrossThreads) {
    constexpr size_t kNumElements = 5;
    std::vector<float> thread1_values(kNumElements);
    std::vector<float> thread2_values(kNumElements);

    std::thread t1([&]() {
        Tensor r1 = Tensor::randn(Shape{kNumElements});
        for (size_t i = 0; i < kNumElements; ++i) {
            thread1_values[i] = r1.data_ptr<float>()[i];
        }
    });
    t1.join();

    std::thread t2([&]() {
        Tensor r2 = Tensor::randn(Shape{kNumElements});
        for (size_t i = 0; i < kNumElements; ++i) {
            thread2_values[i] = r2.data_ptr<float>()[i];
        }
    });
    t2.join();

    // Distinct threads must not produce identical pseudo-random sequences
    EXPECT_NE(thread1_values, thread2_values);
}

// 18. Issue 08: Direct broadcast binary ops (add, sub, mul, div) across differing shapes and dtypes
TEST(AuditRemediationTest2, BinaryBroadcastOpsDirectDispatchAccuracy) {
    Tensor a = Tensor::ones(Shape{1, 4}, DType::Int32);
    Tensor b = Tensor::full(Shape{3, 4}, 2.0f, DType::Float32);

    // add
    Tensor add_res = a + b;
    EXPECT_EQ(add_res.shape(), Shape({3, 4}));
    EXPECT_EQ(add_res.dtype(), DType::Float32);
    for (size_t i = 0; i < add_res.numel(); ++i) {
        EXPECT_FLOAT_EQ(add_res.data_ptr<float>()[i], 3.0f);
    }

    // sub
    Tensor sub_res = b - a;
    EXPECT_EQ(sub_res.shape(), Shape({3, 4}));
    EXPECT_EQ(sub_res.dtype(), DType::Float32);
    for (size_t i = 0; i < sub_res.numel(); ++i) {
        EXPECT_FLOAT_EQ(sub_res.data_ptr<float>()[i], 1.0f);
    }

    // mul
    Tensor mul_res = a * b;
    EXPECT_EQ(mul_res.shape(), Shape({3, 4}));
    EXPECT_EQ(mul_res.dtype(), DType::Float32);
    for (size_t i = 0; i < mul_res.numel(); ++i) {
        EXPECT_FLOAT_EQ(mul_res.data_ptr<float>()[i], 2.0f);
    }

    // div
    Tensor div_res = b / a;
    EXPECT_EQ(div_res.shape(), Shape({3, 4}));
    EXPECT_EQ(div_res.dtype(), DType::Float32);
    for (size_t i = 0; i < div_res.numel(); ++i) {
        EXPECT_FLOAT_EQ(div_res.data_ptr<float>()[i], 2.0f);
    }
}

// 19. Issue 01 (Comprehensive Audit): sum_to_shape zero-dimension broadcast reduction
TEST(AuditRemediationTest2, SumToShapeZeroDimensionReduction) {
    init_autograd();
    Tensor a = Tensor::ones(Shape{1, 5});
    a.set_requires_grad(true);

    Tensor b = Tensor::empty(Shape{0, 5});
    Tensor c = a + b;  // broadcasts to shape {0, 5}
    EXPECT_EQ(c.shape(), Shape({0, 5}));
    EXPECT_EQ(c.numel(), 0);

    Tensor loss = c.sum();
    loss.backward();

    EXPECT_EQ(a.grad().shape(), Shape({1, 5}));
    EXPECT_EQ(a.grad().numel(), 5);
}

// 20. Issue 02 (Comprehensive Audit): AccumulateGrad concurrent backward thread safety
TEST(AuditRemediationTest2, AccumulateGradConcurrentBackwardThreadSafe) {
    init_autograd();
    Tensor shared_w = Tensor::ones(Shape{10, 10});
    shared_w.set_requires_grad(true);

    Tensor y1 = (shared_w * Tensor::ones(Shape{10, 10})).sum();
    Tensor y2 = (shared_w * Tensor::ones(Shape{10, 10})).sum();

    std::thread t1([&]() { y1.backward(); });
    std::thread t2([&]() { y2.backward(); });
    t1.join();
    t2.join();

    EXPECT_TRUE(shared_w.grad().numel() > 0);
    EXPECT_FLOAT_EQ(shared_w.grad().data_ptr<float>()[0], 2.0f);
}

// 21. Issue 05 (Comprehensive Audit): PowBackward zero base and zero exponent
TEST(AuditRemediationTest2, PowBackwardZeroBaseZeroExponent) {
    init_autograd();
    Tensor x({0.0f, 2.0f}, Shape{2});
    x.set_requires_grad(true);

    Tensor y = x.pow(0.0f);
    Tensor loss = y.sum();
    loss.backward();

    const float* grad_ptr = x.grad().data_ptr<float>();
    EXPECT_FALSE(std::isnan(grad_ptr[0]));
    EXPECT_FLOAT_EQ(grad_ptr[0], 0.0f);
    EXPECT_FLOAT_EQ(grad_ptr[1], 0.0f);
}

// 22. Issue 06 (Comprehensive Audit): promote_types(Int64, Float32) yields Float64
TEST(AuditRemediationTest2, PromoteTypesInt64Float32YieldsFloat64) {
    EXPECT_EQ(promote_types(DType::Int64, DType::Float32), DType::Float64);
    EXPECT_EQ(promote_types(DType::Float32, DType::Int64), DType::Float64);
}
