#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "backend/cpu_backend.hpp"

using namespace helix;

// Helper to fill vector with random floats
void fill_random(std::vector<float>& vec) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& val : vec) {
        val = dis(gen);
    }
}

// Helper to verify correctness
bool verify_results(const float* ref, const float* test, size_t size, float tol = 1e-3f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(ref[i] - test[i]);
        if (diff > tol) {
            std::cout << "Mismatch at index " << i << ": expected " << std::scientific << std::setprecision(8) << ref[i]
                      << ", got " << test[i] << ", diff: " << diff << std::endl;
            return false;
        }
    }
    return true;
}

struct BenchResult {
    double time_ms;
    double gflops;
    double speedup;
    bool correct;
};

void run_benchmark_for_size(size_t M, size_t K, size_t N) {
    std::cout << "Benchmarking Matrix Size: " << M << " x " << K << " x " << N << std::endl;

    std::vector<float> A(M * K);
    std::vector<float> B_T(N * K);  // B transposed
    std::vector<float> C_ref(M * N, 0.0f);
    std::vector<float> C_tiled(M * N, 0.0f);
    std::vector<float> C_avx2(M * N, 0.0f);

    fill_random(A);
    fill_random(B_T);

    double ops = 2.0 * M * N * K;

    // 1. Naive
    // Warm up
    CPUBackend::matmul(A.data(), B_T.data(), C_ref.data(), M, K, N);  // Currently points to naive

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = M >= 1000 ? 5 : 20;
    for (int i = 0; i < iterations; ++i) {
        // We will call a dedicated matmul_naive in CPUBackend to bypass routing
        CPUBackend::matmul_naive(A.data(), B_T.data(), C_ref.data(), M, K, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gflops_naive = (ops / (time_naive / 1000.0)) / 1e9;

    // 2. Tiled
    CPUBackend::matmul_tiled(A.data(), B_T.data(), C_tiled.data(), M, K, N);
    bool tiled_ok = verify_results(C_ref.data(), C_tiled.data(), M * N);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        CPUBackend::matmul_tiled(A.data(), B_T.data(), C_tiled.data(), M, K, N);
    }
    end = std::chrono::high_resolution_clock::now();
    double time_tiled = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    double gflops_tiled = (ops / (time_tiled / 1000.0)) / 1e9;
    double speedup_tiled = time_naive / time_tiled;

    // 3. AVX2
    double time_avx2 = 0.0;
    double gflops_avx2 = 0.0;
    double speedup_avx2 = 0.0;
    bool avx2_ok = false;
#if defined(__AVX2__)
    CPUBackend::matmul_avx2(A.data(), B_T.data(), C_avx2.data(), M, K, N);
    avx2_ok = verify_results(C_ref.data(), C_avx2.data(), M * N);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        CPUBackend::matmul_avx2(A.data(), B_T.data(), C_avx2.data(), M, K, N);
    }
    end = std::chrono::high_resolution_clock::now();
    time_avx2 = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    gflops_avx2 = (ops / (time_avx2 / 1000.0)) / 1e9;
    speedup_avx2 = time_naive / time_avx2;
#endif

    // Print results
    std::cout << std::left << std::setw(15) << "Kernel" << std::setw(15) << "Time (ms)" << std::setw(15) << "GFLOPS"
              << std::setw(15) << "Speedup" << std::setw(10) << "Correct" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    std::cout << std::left << std::setw(15) << "Naive" << std::setw(15) << time_naive << std::setw(15) << gflops_naive
              << std::setw(15) << "1.0x (Ref)" << std::setw(10) << "OK" << std::endl;

    std::cout << std::left << std::setw(15) << "Tiled" << std::setw(15) << time_tiled << std::setw(15) << gflops_tiled
              << std::setw(15) << (std::to_string(speedup_tiled) + "x") << std::setw(10) << (tiled_ok ? "OK" : "FAIL")
              << std::endl;

#if defined(__AVX2__)
    std::cout << std::left << std::setw(15) << "AVX2" << std::setw(15) << time_avx2 << std::setw(15) << gflops_avx2
              << std::setw(15) << (std::to_string(speedup_avx2) + "x") << std::setw(10) << (avx2_ok ? "OK" : "FAIL")
              << std::endl;
#else
    std::cout << std::left << std::setw(15) << "AVX2" << std::setw(15) << "N/A" << std::setw(15) << "N/A"
              << std::setw(15) << "N/A" << std::setw(10) << "N/A" << std::endl;
#endif
    std::cout << std::endl;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "          HELIX MATMUL BENCHMARK            " << std::endl;
    std::cout << "============================================" << std::endl;

#if defined(__AVX2__)
    std::cout << "AVX2 Instruction Set: DETECTED" << std::endl;
#else
    std::cout << "AVX2 Instruction Set: NOT DETECTED" << std::endl;
#endif

#if defined(__FMA__)
    std::cout << "FMA Instruction Set:  DETECTED" << std::endl;
#else
    std::cout << "FMA Instruction Set:  NOT DETECTED" << std::endl;
#endif

#if defined(_OPENMP)
    std::cout << "OpenMP Multi-threading: DETECTED" << std::endl;
#else
    std::cout << "OpenMP Multi-threading: NOT DETECTED" << std::endl;
#endif
    std::cout << "============================================" << std::endl << std::endl;

    std::vector<size_t> sizes = {127, 128, 255, 256, 511, 512, 1023, 1024};
    for (size_t n : sizes) {
        run_benchmark_for_size(n, n, n);
    }

    return 0;
}
