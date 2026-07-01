#include <iomanip>
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

    // Công thức tính số phép toán dấu phẩy động thống nhất: 2 * M * K * N
    double ops = 2.0 * M * K * N;

    // Strategies to benchmark
    // Strategies to benchmark
    std::vector<std::pair<MatMulStrategy, std::string>> strategies = {
        {MatMulStrategy::Naive, "Naive"},
        {MatMulStrategy::Blocked, "Blocked"},
        {MatMulStrategy::AVX2, "AVX2"},
        {MatMulStrategy::OpenMP, "OpenMP"}
    };

    std::vector<BenchmarkResult> results;
    constexpr double OPENBLAS_GFLOPS = 240.0;  // Theoretical/Baseline OpenBLAS performance

    for (size_t i = 0; i < strategies.size(); ++i) {
        auto strategy = strategies[i].first;
        auto name_suffix = strategies[i].second;

        auto fn = [&]() { CPUBackend::matmul(A.data(), B_T.data(), C.data(), M, K, N, strategy); };
        std::string name = name_suffix + " " + std::to_string(size) + "x" + std::to_string(size);

        // 5 warmups, 30 iterations for stability
        BenchmarkResult res = BenchmarkRunner::run(name, fn, 30, 5, ops);
        BenchmarkReporter::print_result(res);

        // Print Efficiency vs OpenBLAS
        double efficiency = (res.gflops / OPENBLAS_GFLOPS) * 100.0;
        std::cout << "  -> Efficiency vs OpenBLAS (240 GFLOPS): " << std::fixed << std::setprecision(2) << efficiency
                  << "%" << std::endl;

        // Calculate and Print Speedup
        if (i > 0) {
            double speedup = results[i - 1].median_ms / res.median_ms;
            std::cout << "  -> Speedup vs " << strategies[i - 1].second << ": " << std::fixed << std::setprecision(2)
                      << speedup << "x" << std::endl;
        }

        if (i == strategies.size() - 1) {
            // Compare OpenMP vs Naive
            double speedup_total = results[0].median_ms / res.median_ms;
            std::cout << "  -> Total Speedup (OpenMP vs Naive): " << std::fixed << std::setprecision(2) << speedup_total
                      << "x" << std::endl;
        }

        std::cout << std::endl;
        results.push_back(res);
    }

    std::cout << "--------------------------------------------------------" << std::endl;

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
