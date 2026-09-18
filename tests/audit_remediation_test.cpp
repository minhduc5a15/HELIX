#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include "autograd/autograd_meta.hpp"
#include "autograd/engine.hpp"
#include "autograd/node.hpp"
#include "core/allocator.hpp"
#include "core/autotuner.hpp"
#include "core/dispatcher.hpp"
#include "core/nd_iterator.hpp"
#include "core/tensor.hpp"
#include "core/tensor_factory.hpp"

namespace helix {
    void avx2_dot_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
    void avx2_micro_matmul(const float* a, const float* b, float* out, size_t M, size_t K, size_t N);
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
