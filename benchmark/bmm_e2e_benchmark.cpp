#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "autograd/engine.hpp"
#include "backend/matmul/bmm_internal.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "core/cpu_utils.hpp"
#include "core/tensor.hpp"

using namespace helix;
using namespace helix::benchmark;

struct E2EWorkload {
    size_t B;
    size_t M;
    size_t K;
    size_t N;
    std::string category;
    size_t default_iterations;
    size_t default_warmups;
};

struct E2ERecord {
    size_t B;
    size_t M;
    size_t K;
    size_t N;
    size_t threads;
    std::string category;
    std::string dispatched_policy;
    double forward_only_ms;
    double forward_backward_ms;
    double backward_ratio;
    double forward_gflops;
};

static void export_e2e_csv(const std::vector<E2ERecord>& records, const std::string& filepath) {
    const std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        return;
    }

    file << "B,M,K,N,threads,category,dispatched_policy,forward_only_ms,forward_backward_ms,backward_ratio,forward_"
            "gflops\n";
    for (const auto& r : records) {
        file << r.B << "," << r.M << "," << r.K << "," << r.N << "," << r.threads << ",\"" << r.category << "\","
             << r.dispatched_policy << "," << std::fixed << std::setprecision(6) << r.forward_only_ms << ","
             << r.forward_backward_ms << "," << std::setprecision(2) << r.backward_ratio << "," << std::setprecision(4)
             << r.forward_gflops << "\n";
    }
    file.close();
    std::cout << "Successfully exported E2E benchmark results to: " << filepath << "\n";
}

int main(int argc, char** argv) {
    init_autograd();
    Tensor::manual_seed(42);

    bool full_mode = true;
    std::string output_path = "output/bmm_benchmark_e2e.csv";
    int custom_iterations = -1;
    int custom_warmups = -1;
    size_t threads = 0;

#if defined(_OPENMP)
    threads = static_cast<size_t>(omp_get_max_threads());
#else
    threads = 1;
#endif

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            full_mode = false;
        } else if (arg == "--full") {
            full_mode = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            custom_iterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            custom_warmups = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = static_cast<size_t>(std::stoul(argv[++i]));
#if defined(_OPENMP)
            omp_set_num_threads(static_cast<int>(threads));
#endif
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bmm_e2e_benchmark [options]\n"
                      << "Options:\n"
                      << "  --quick               Run only initial 3 boundary workloads\n"
                      << "  --full                Run all 6 representative workloads (default)\n"
                      << "  --output <path>       CSV output filepath (default: output/bmm_benchmark_e2e.csv)\n"
                      << "  --iterations <N>      Override iteration count\n"
                      << "  --warmup <N>          Override warmup count\n"
                      << "  --threads <N>         Override OpenMP thread count\n";
            return 0;
        }
    }

#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#pragma omp parallel
    { (void)omp_get_thread_num(); }
#endif

    std::cout << "==============================================================================\n"
              << " HELIX Tensor BMM End-to-End Performance Suite (Phase 1.1)\n"
              << "==============================================================================\n"
              << "Hardware / Environment:\n"
              << "  OpenMP Threads    : " << threads << "\n"
              << "  AVX2 / FMA Support: " << (supports_avx2_backend() ? "YES" : "NO") << "\n"
              << "  Execution Mode    : " << (full_mode ? "Full (6 workloads)" : "Quick (3 workloads)") << "\n"
              << "  Output CSV Target : " << output_path << "\n"
              << "------------------------------------------------------------------------------\n";

    std::vector<E2EWorkload> workloads = {
        {4, 16, 16, 16, "Below threshold (< 32K MACs)", 50, 10},
        {8, 16, 16, 16, "At threshold (= 32K MACs)", 50, 10},
        {4, 20, 20, 20, "Known outlier (< 32K MACs)", 50, 10},
        {4, 128, 64, 128, "Attention-like rectangular", 25, 5},
        {4, 256, 256, 256, "Standard compute-intensive", 15, 3},
        {2, 512, 512, 512, "Large GEMM (B < threads)", 5, 2},
    };

    if (!full_mode) {
        workloads.resize(3);
    }

    std::vector<E2ERecord> records;

    for (const auto& wl : workloads) {
        const size_t B = wl.B;
        const size_t M = wl.M;
        const size_t K = wl.K;
        const size_t N = wl.N;

        const size_t iters = (custom_iterations > 0) ? static_cast<size_t>(custom_iterations) : wl.default_iterations;
        const size_t warmups = (custom_warmups >= 0) ? static_cast<size_t>(custom_warmups) : wl.default_warmups;
        const double fwd_flops =
            2.0 * static_cast<double>(B) * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);

        // Dispatched policy
        auto policy = internal::choose_bmm_execution_policy(B, M, K, N, threads);
        std::string policy_str =
            (policy == internal::BmmExecutionPolicy::ParallelBatch) ? "ParallelBatch" : "SerialAuto";

        std::cout << "\n>>> Workload: [" << B << ", " << M << "x" << K << "x" << N << "] " << wl.category
                  << " | Policy: " << policy_str << "\n";

        // 1. Forward-Only (Inference under no_grad)
        Tensor a_eval = Tensor::randn(Shape({B, M, K}));
        Tensor b_eval = Tensor::randn(Shape({B, K, N}));

        auto fn_forward = [&]() {
            no_grad guard;
            Tensor c = a_eval.bmm(b_eval);
            (void)c;
        };

        std::string fw_name = "E2E_Forward [" + std::to_string(B) + "x" + std::to_string(M) + "x" + std::to_string(K) +
                              "x" + std::to_string(N) + "]";
        BenchmarkResult res_fw = BenchmarkRunner::run(fw_name, fn_forward, iters, warmups, fwd_flops);

        // 2. Forward + Backward (Training with fresh graph & gradient reset)
        Tensor a_train = Tensor::randn(Shape({B, M, K}));
        Tensor b_train = Tensor::randn(Shape({B, K, N}));
        a_train.set_requires_grad(true);
        b_train.set_requires_grad(true);

        auto fn_forward_backward = [&]() {
            if (a_train.has_grad()) a_train.grad().zero_();
            if (b_train.has_grad()) b_train.grad().zero_();

            Tensor c = a_train.bmm(b_train);
            Tensor loss = c.sum();
            loss.backward();
        };

        std::string fw_bw_name = "E2E_FW+BW [" + std::to_string(B) + "x" + std::to_string(M) + "x" + std::to_string(K) +
                                 "x" + std::to_string(N) + "]";
        // Forward + backward computes forward GEMM + 2 backward GEMMs (~3x forward operations)
        const double total_fw_bw_flops = fwd_flops * 3.0;
        BenchmarkResult res_fw_bw =
            BenchmarkRunner::run(fw_bw_name, fn_forward_backward, iters, warmups, total_fw_bw_flops);

        double backward_ratio = (res_fw.median_ms > 0.0) ? (res_fw_bw.median_ms / res_fw.median_ms) : 0.0;
        double fwd_gflops = (res_fw.median_ms > 0.0) ? (fwd_flops / (res_fw.median_ms * 1e6)) : 0.0;

        std::cout << std::fixed << std::setprecision(4) << "  forward_only_ms     : " << res_fw.median_ms << " ms ("
                  << fwd_gflops << " GFLOPS)\n"
                  << "  forward_backward_ms : " << res_fw_bw.median_ms << " ms (" << std::setprecision(2)
                  << backward_ratio << "x forward)\n";

        E2ERecord rec;
        rec.B = B;
        rec.M = M;
        rec.K = K;
        rec.N = N;
        rec.threads = threads;
        rec.category = wl.category;
        rec.dispatched_policy = policy_str;
        rec.forward_only_ms = res_fw.median_ms;
        rec.forward_backward_ms = res_fw_bw.median_ms;
        rec.backward_ratio = backward_ratio;
        rec.forward_gflops = fwd_gflops;
        records.push_back(rec);
    }

    std::cout << "\n==============================================================================\n"
              << " E2E Summary Results\n"
              << "==============================================================================\n"
              << std::setw(22) << std::left << "Shape [B,MxKxN]" << std::setw(15) << std::left << "Policy"
              << std::setw(15) << std::right << "Forward (ms)" << std::setw(16) << std::right << "FW+BW (ms)"
              << std::setw(12) << std::right << "Ratio" << std::setw(15) << std::right << "FW GFLOPS\n"
              << "------------------------------------------------------------------------------\n";

    for (const auto& r : records) {
        std::string shape_str = "[" + std::to_string(r.B) + ", " + std::to_string(r.M) + "x" + std::to_string(r.K) +
                                "x" + std::to_string(r.N) + "]";
        std::cout << std::setw(22) << std::left << shape_str << std::setw(15) << std::left << r.dispatched_policy
                  << std::setw(15) << std::right << std::fixed << std::setprecision(4) << r.forward_only_ms
                  << std::setw(16) << std::right << r.forward_backward_ms << std::setw(11) << std::right
                  << std::setprecision(2) << r.backward_ratio << "x" << std::setw(15) << std::right
                  << std::setprecision(2) << r.forward_gflops << "\n";
    }
    std::cout << "==============================================================================\n";

    export_e2e_csv(records, output_path);
    return 0;
}
