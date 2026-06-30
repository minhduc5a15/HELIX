#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "backend/cpu_backend.hpp"
#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"

using namespace helix;
using namespace helix::benchmark;

static void fill_random(std::vector<float>& vec) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

static std::vector<BenchmarkResult> run_matmul_benchmark(size_t size) {
    size_t M = size;
    size_t K = size;
    size_t N = size;

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    double ops = 2.0 * M * K * N;

    // Strategies to benchmark
    std::vector<std::pair<MatMulStrategy, std::string>> strategies = {
        {MatMulStrategy::Naive, "Naive"}, {MatMulStrategy::Blocked, "Blocked"}
    };

    std::vector<BenchmarkResult> results;

    for (const auto& [strategy, name_suffix] : strategies) {
        auto fn = [&]() { CPUBackend::matmul(A.data(), B_T.data(), C.data(), M, K, N, strategy); };
        std::string name = name_suffix + " " + std::to_string(size) + "x" + std::to_string(size);
        BenchmarkResult res = BenchmarkRunner::run(name, fn, 10, 3, ops);
        BenchmarkReporter::print_result(res);
        results.push_back(res);
    }

    // Print comparison
    if (results.size() >= 2) {
        BenchmarkReporter::print_comparison(results[0], results[1]);
    }

    return results;
}

int main() {
    BenchmarkReporter::print_header("Matrix Multiplication Benchmark");

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};
    std::vector<BenchmarkResult> all_results;
    for (size_t size : sizes) {
        auto res = run_matmul_benchmark(size);
        all_results.insert(all_results.end(), res.begin(), res.end());
    }

    BenchmarkReporter::export_csv(all_results, "output/matmul_benchmark.csv");

    BenchmarkReporter::print_footer();
    return 0;
}
