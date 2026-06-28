#include <benchmark/benchmark.h>
#include <vector>
#include <random>

#include "backend/cpu_backend.hpp"

using namespace helix;

static void fill_random(std::vector<float>& vec) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

static void BM_MatMul_Naive(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    for (auto _ : state) {
        CPUBackend::matmul_naive(A.data(), B_T.data(), C.data(), M, K, N);
        benchmark::DoNotOptimize(C.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        (state.iterations() * 2.0 * M * K * N) / 1e9, 
        benchmark::Counter::kIsRate
    );
}

static void BM_MatMul_Tiled(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    for (auto _ : state) {
        CPUBackend::matmul_tiled(A.data(), B_T.data(), C.data(), M, K, N);
        benchmark::DoNotOptimize(C.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        (state.iterations() * 2.0 * M * K * N) / 1e9, 
        benchmark::Counter::kIsRate
    );
}

static void BM_MatMul_AVX2_SingleThread(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    for (auto _ : state) {
        CPUBackend::matmul_avx2(A.data(), B_T.data(), C.data(), M, K, N);
        benchmark::DoNotOptimize(C.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        (state.iterations() * 2.0 * M * K * N) / 1e9, 
        benchmark::Counter::kIsRate
    );
}

static void BM_MatMul_Optimized_OpenMP(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    for (auto _ : state) {
        CPUBackend::matmul(A.data(), B_T.data(), C.data(), M, K, N);
        benchmark::DoNotOptimize(C.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        (state.iterations() * 2.0 * M * K * N) / 1e9, 
        benchmark::Counter::kIsRate
    );
}

static void CustomArguments(benchmark::internal::Benchmark* b) {
    b->Args({127, 127, 127});
    b->Args({511, 511, 511});
    b->Args({1023, 1023, 1023});
}

BENCHMARK(BM_MatMul_Naive)->Apply(CustomArguments)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MatMul_Tiled)->Apply(CustomArguments)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MatMul_AVX2_SingleThread)->Apply(CustomArguments)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MatMul_Optimized_OpenMP)->Apply(CustomArguments)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
