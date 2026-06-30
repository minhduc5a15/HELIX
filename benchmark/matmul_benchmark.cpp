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

void run_matmul_benchmark(size_t size) {
    size_t M = size;
    size_t K = size;
    size_t N = size;

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);
    std::vector<float> C(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    double ops = 2.0 * M * K * N;

    // We only benchmark Naive for Week 1 baseline
    auto fn_naive = [&]() { CPUBackend::matmul_naive(A.data(), B_T.data(), C.data(), M, K, N); };

    /*
    auto fn_tiled = [&]() {
        CPUBackend::matmul_tiled(A.data(), B_T.data(), C.data(), M, K, N);
    };

    auto fn_avx2 = [&]() {
        CPUBackend::matmul_avx2(A.data(), B_T.data(), C.data(), M, K, N);
    };

    auto fn_openmp = [&]() {
        CPUBackend::matmul(A.data(), B_T.data(), C.data(), M, K, N);
    };
    */

    std::string name = "Naive " + std::to_string(size) + "x" + std::to_string(size);
    BenchmarkResult res_naive = BenchmarkRunner::run(name, fn_naive, 10, 3, ops);
    BenchmarkReporter::print_result(res_naive);

    /*
    std::string name_tiled = "Blocked " + std::to_string(size) + "x" + std::to_string(size);
    BenchmarkResult res_tiled = BenchmarkRunner::run(name_tiled, fn_tiled, 10, 3, ops);
    BenchmarkReporter::print_result(res_tiled);

    std::string name_avx2 = "AVX2 " + std::to_string(size) + "x" + std::to_string(size);
    BenchmarkResult res_avx2 = BenchmarkRunner::run(name_avx2, fn_avx2, 10, 3, ops);
    BenchmarkReporter::print_result(res_avx2);

    std::string name_omp = "OpenMP " + std::to_string(size) + "x" + std::to_string(size);
    BenchmarkResult res_omp = BenchmarkRunner::run(name_omp, fn_openmp, 10, 3, ops);
    BenchmarkReporter::print_result(res_omp);
    */
}

int main() {
    BenchmarkReporter::print_header("Matrix Multiplication Benchmark");

    std::vector<size_t> sizes = {64, 128, 256, 512, 1024};
    for (size_t size : sizes) {
        run_matmul_benchmark(size);
    }

    BenchmarkReporter::print_footer();
    return 0;
}
