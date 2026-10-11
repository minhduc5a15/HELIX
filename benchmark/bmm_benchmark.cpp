#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "backend/cpu_backend.hpp"
#include "backend/matmul/bmm_internal.hpp"
#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "core/cpu_utils.hpp"

using namespace helix;
using namespace helix::benchmark;

struct BmmWorkload {
    size_t B;
    size_t M;
    size_t K;
    size_t N;
    std::string category;
};

struct BmmBenchmarkRecord {
    size_t B;
    size_t M;
    size_t K;
    size_t N;
    size_t threads;
    std::string mode;
    double median_ms;
    double average_ms;
    double stddev_ms;
    double gflops;
    double speedup_vs_serial_auto;
    size_t matrix_work;
    size_t effective_batch_parallelism;
    size_t matrix_footprint_bytes;
    size_t aggregate_active_footprint_bytes;
    std::string inner_kernel;
};

static void fill_deterministic(std::vector<float>& vec, float offset) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = static_cast<float>(static_cast<int>((i + 3) % 19) - 9) * 0.125f + offset;
    }
}

static void export_bmm_csv(const std::vector<BmmBenchmarkRecord>& records, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing\n";
        return;
    }

    file << "B,M,K,N,threads,mode,median_ms,average_ms,stddev_ms,gflops,speedup_vs_serial_auto,"
         << "matrix_work,effective_batch_parallelism,matrix_footprint_bytes,aggregate_active_footprint_bytes,inner_"
            "kernel\n";

    for (const auto& r : records) {
        file << r.B << "," << r.M << "," << r.K << "," << r.N << "," << r.threads << "," << r.mode << "," << std::fixed
             << std::setprecision(4) << r.median_ms << "," << r.average_ms << "," << r.stddev_ms << "," << r.gflops
             << "," << r.speedup_vs_serial_auto << "," << r.matrix_work << "," << r.effective_batch_parallelism << ","
             << r.matrix_footprint_bytes << "," << r.aggregate_active_footprint_bytes << "," << r.inner_kernel << "\n";
    }
    file.close();
    std::cout << "Successfully exported benchmark results to " << filepath << "\n";
}

static void benchmark_workload(
    const BmmWorkload& wl,
    size_t threads,
    size_t warmups,
    size_t iterations,
    const std::string& inner_kernel_name,
    std::vector<BmmBenchmarkRecord>& records
) {
#if defined(_OPENMP)
    omp_set_num_threads(static_cast<int>(threads));
#endif

    const size_t B = wl.B;
    const size_t M = wl.M;
    const size_t K = wl.K;
    const size_t N = wl.N;

    std::vector<float> a(B * M * K);
    std::vector<float> b(B * K * N);
    fill_deterministic(a, 0.0f);
    fill_deterministic(b, 0.05f);

    const double operations =
        2.0 * static_cast<double>(B) * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    const size_t matrix_work = M * K * N;
    const size_t effective_batch_parallelism = std::min(B, threads);
    const size_t matrix_footprint_bytes = sizeof(float) * (M * K + K * N + M * N);
    const size_t aggregate_active_footprint_bytes = effective_batch_parallelism * matrix_footprint_bytes;

    // Validate reference output and candidate output equality
    std::vector<float> ref_out(B * M * N, 0.0f);
    std::vector<float> cand_out(B * M * N, 0.0f);

    internal::bmm_serial_batches(a.data(), b.data(), ref_out.data(), B, M, K, N, MatMulStrategy::Auto);
    internal::bmm_parallel_batches_float(a.data(), b.data(), cand_out.data(), B, M, K, N);

    for (size_t i = 0; i < ref_out.size(); ++i) {
        if (std::abs(ref_out[i] - cand_out[i]) > 1e-3f) {
            std::cerr << "VALIDATION FAILURE in workload [" << B << ", " << M << ", " << K << ", " << N
                      << "]: diff=" << std::abs(ref_out[i] - cand_out[i]) << " at idx " << i << "\n";
            return;
        }
    }

    struct ModeDef {
        std::string name;
        std::function<void()> fn;
    };

    std::vector<float> out(B * M * N, 0.0f);

    std::vector<ModeDef> modes = {
        {"serial_auto",
         [&]() { internal::bmm_serial_batches(a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::Auto); }},
        {"serial_avx2",
         [&]() { internal::bmm_serial_batches(a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::AVX2); }},
        {"serial_openmp",
         [&]() { internal::bmm_serial_batches(a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::OpenMP); }},
        {"parallel_batch_serial_inner",
         [&]() { internal::bmm_parallel_batches_float(a.data(), b.data(), out.data(), B, M, K, N); }}
    };

    double serial_auto_median = 1.0;

    for (size_t i = 0; i < modes.size(); ++i) {
        std::string test_name = modes[i].name + " [B=" + std::to_string(B) + " " + std::to_string(M) + "x" +
                                std::to_string(K) + "x" + std::to_string(N) + "] T=" + std::to_string(threads);

        BenchmarkResult res = BenchmarkRunner::run(test_name, modes[i].fn, iterations, warmups, operations);

        if (i == 0) {
            serial_auto_median = res.median_ms;
        }

        const double speedup =
            (serial_auto_median > 0.0 && res.median_ms > 0.0) ? (serial_auto_median / res.median_ms) : 1.0;

        BmmBenchmarkRecord rec;
        rec.B = B;
        rec.M = M;
        rec.K = K;
        rec.N = N;
        rec.threads = threads;
        rec.mode = modes[i].name;
        rec.median_ms = res.median_ms;
        rec.average_ms = res.average_ms;
        rec.stddev_ms = res.stddev_ms;
        rec.gflops = res.gflops;
        rec.speedup_vs_serial_auto = speedup;
        rec.matrix_work = matrix_work;
        rec.effective_batch_parallelism = effective_batch_parallelism;
        rec.matrix_footprint_bytes = matrix_footprint_bytes;
        rec.aggregate_active_footprint_bytes = aggregate_active_footprint_bytes;
        rec.inner_kernel = inner_kernel_name;

        records.push_back(rec);
    }
}

int main(int argc, char** argv) {
    BenchmarkReporter::print_header("HELIX Float32 BMM Scheduling Policy Benchmark");

    bool run_refined = false;
    std::string output_path = "output/bmm_benchmark_scout.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--refined") {
            run_refined = true;
            output_path = "output/bmm_benchmark_refined.csv";
        } else if (arg == "--scout") {
            run_refined = false;
            output_path = "output/bmm_benchmark_scout.csv";
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        }
    }

    const std::string inner_kernel = supports_avx2_backend() ? "avx2" : "blocked";
    std::cout << "Inner kernel used: " << inner_kernel << "\n";

    // Determine thread counts to benchmark
    int max_threads = 1;
#if defined(_OPENMP)
    max_threads = omp_get_max_threads();
#endif

    std::vector<size_t> thread_counts;
    auto add_threads = [&](size_t t) {
        if (t >= 1 && t <= static_cast<size_t>(max_threads)) {
            if (std::find(thread_counts.begin(), thread_counts.end(), t) == thread_counts.end()) {
                thread_counts.push_back(t);
            }
        }
    };

    add_threads(1);
    add_threads(2);
    add_threads(4);
    add_threads(6);  // Physical core count on standard 6-core dev environment
    add_threads(static_cast<size_t>(max_threads));

    std::sort(thread_counts.begin(), thread_counts.end());

    std::cout << "Evaluating thread configurations: ";
    for (size_t t : thread_counts) std::cout << t << " ";
    std::cout << "\n";

    std::vector<BmmWorkload> workloads;

    if (!run_refined) {
        // --- STAGE A: SCOUT WORKLOADS ---
        std::cout << "Running Stage A: Scout Benchmark Suite...\n";

        // 1. Tiny GEMM: 16x16x16, 32x32x32 with varied batches
        for (size_t B : {8, 16, 32, 64}) {
            workloads.push_back({B, 16, 16, 16, "tiny_16"});
            workloads.push_back({B, 32, 32, 32, "tiny_32"});
        }

        // 2. Small/Medium GEMM: 64x64x64, 128x128x128 with varied batches
        for (size_t B : {2, 4, 8, 16, 32}) {
            workloads.push_back({B, 64, 64, 64, "small_64"});
            workloads.push_back({B, 128, 128, 128, "medium_128"});
        }

        // 3. Large GEMM: 256x256x256 with varied batches
        for (size_t B : {1, 2, 4, 8, 16}) {
            workloads.push_back({B, 256, 256, 256, "large_256"});
        }

        // 4. Very Large GEMM: 512x512x512
        for (size_t B : {1, 2, 4}) {
            workloads.push_back({B, 512, 512, 512, "vlarge_512"});
        }

        // 5. Rectangular Attention-Like Workloads
        for (size_t B : {2, 4, 8, 16}) {
            workloads.push_back({B, 128, 64, 128, "attn_128_64_128"});
            workloads.push_back({B, 128, 128, 64, "attn_128_128_64"});
            workloads.push_back({B, 256, 64, 256, "attn_256_64_256"});
            workloads.push_back({B, 64, 128, 64, "attn_64_128_64"});
        }
    } else {
        // --- STAGE B: REFINED WORKLOADS ---
        std::cout << "Running Stage B: Refined Crossover Benchmark Suite...\n";

        // Focused boundary exploration around crossover regions
        for (size_t B : {2, 3, 4, 5, 6, 8, 12, 16}) {
            workloads.push_back({B, 64, 64, 64, "refined_64"});
            workloads.push_back({B, 96, 96, 96, "refined_96"});
            workloads.push_back({B, 128, 128, 128, "refined_128"});
            workloads.push_back({B, 160, 160, 160, "refined_160"});
            workloads.push_back({B, 192, 192, 192, "refined_192"});
        }
    }

    const size_t warmups = run_refined ? 4 : 2;
    const size_t iterations = run_refined ? 15 : 7;

    std::vector<BmmBenchmarkRecord> all_records;

    size_t current = 0;
    const size_t total = workloads.size() * thread_counts.size();

    for (const auto& wl : workloads) {
        for (size_t threads : thread_counts) {
            ++current;
            if (current % 10 == 0 || current == total) {
                std::cout << "Progress: " << current << " / " << total << " configurations (" << (current * 100 / total)
                          << "%)...\r" << std::flush;
            }
            benchmark_workload(wl, threads, warmups, iterations, inner_kernel, all_records);
        }
    }
    std::cout << "\nBenchmark run complete! Total data points collected: " << all_records.size() << "\n";

    export_bmm_csv(all_records, output_path);
    BenchmarkReporter::print_footer();

    return 0;
}
