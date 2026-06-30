#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <stdexcept>

#include "benchmark/benchmark_runner.hpp"
#include "benchmark/timer.hpp"

using namespace helix::benchmark;

// ---------------------------------------------------------
// 1. Timer Tests (Happy Path & Edge Cases)
// ---------------------------------------------------------

TEST(BenchmarkTest, TimerHappyPath) {
    Timer t;
    t.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    t.stop();

    double ms = t.elapsed_ms();
    EXPECT_GE(ms, 45.0);
    EXPECT_LE(ms, 250.0);
}

TEST(BenchmarkTest, TimerEdgeCase_InstantStop) {
    Timer t;
    t.start();
    t.stop();

    EXPECT_GE(t.elapsed_ns(), 0.0);
    EXPECT_GE(t.elapsed_us(), 0.0);
    EXPECT_GE(t.elapsed_ms(), 0.0);
}

// ---------------------------------------------------------
// 2. BenchmarkRunner Tests (Happy Path, Edge Cases, Error Handling, Stress)
// ---------------------------------------------------------

TEST(BenchmarkTest, RunnerHappyPath) {
    int count = 0;
    auto fn = [&]() { count++; };

    BenchmarkResult res = BenchmarkRunner::run("HappyPath", fn, 10, 5, 100.0);

    EXPECT_EQ(res.name, "HappyPath");
    EXPECT_EQ(res.iterations, 10);
    EXPECT_EQ(count, 15);  // 5 warmup + 10 iterations
    EXPECT_GE(res.average_ms, 0.0);
    EXPECT_GE(res.maximum_ms, res.minimum_ms);
    EXPECT_GE(res.gflops, 0.0);
}

TEST(BenchmarkTest, RunnerEdgeCase_ZeroWarmup) {
    int count = 0;
    auto fn = [&]() { count++; };

    BenchmarkResult res = BenchmarkRunner::run("ZeroWarmup", fn, 10, 0, 0.0);

    EXPECT_EQ(res.iterations, 10);
    EXPECT_EQ(count, 10); 
    EXPECT_EQ(res.gflops, 0.0);
}

TEST(BenchmarkTest, RunnerEdgeCase_OneIteration) {
    int count = 0;
    auto fn = [&]() { count++; };

    BenchmarkResult res = BenchmarkRunner::run("OneIteration", fn, 1, 0, 10.0);

    EXPECT_EQ(res.iterations, 1);
    EXPECT_EQ(count, 1);
    EXPECT_EQ(res.minimum_ms, res.maximum_ms);
    EXPECT_EQ(res.average_ms, res.maximum_ms);
    EXPECT_EQ(res.stddev_ms, 0.0); // Variance of 1 item is 0
}

TEST(BenchmarkTest, RunnerErrorHandling_ZeroIterations) {
    auto fn = []() {};
    
    // Iterations = 0 should throw std::invalid_argument
    EXPECT_THROW(BenchmarkRunner::run("ZeroIters", fn, 0, 5, 10.0), std::invalid_argument);
}

TEST(BenchmarkTest, RunnerErrorHandling_FunctionThrows) {
    auto fn = []() { throw std::runtime_error("Simulated crash"); };
    
    // Runner should propagate the exception thrown by fn
    EXPECT_THROW(BenchmarkRunner::run("Crash", fn, 10, 5, 10.0), std::runtime_error);
}

TEST(BenchmarkTest, RunnerStress_HighVolume) {
    int count = 0;
    auto fn = [&]() { count++; };

    // Stress testing the runner with a very large number of iterations
    BenchmarkResult res = BenchmarkRunner::run("Stress", fn, 100000, 1000, 0.0);

    EXPECT_EQ(res.iterations, 100000);
    EXPECT_EQ(count, 101000);
}
