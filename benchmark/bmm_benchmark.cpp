#include <algorithm>
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

    file << "B,M,K,N,threads,mode,median_ms,average_ms,stddev_ms,gflops,speedup_vs_serial_auto,"
         << "matrix_work,effective_batch_parallelism,matrix_footprint_bytes,aggregate_active_footprint_bytes,inner_"
            "kernel\n";

    for (const auto& r : records) {
        file << r.B << "," << r.M << "," << r.K << "," << r.N << "," << r.threads << "," << r.mode << "," << std::fixed
             << std::setprecision(6) << r.median_ms << "," << r.average_ms << "," << r.stddev_ms << "," << r.gflops
             << "," << std::setprecision(4) << r.speedup_vs_serial_auto << "," << r.matrix_work << ","
             << r.effective_batch_parallelism << "," << r.matrix_footprint_bytes << ","
             << r.aggregate_active_footprint_bytes << "," << r.inner_kernel << "\n";
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
// Warm up thread team to eliminate first-touch OS thread pool spawn latency
#pragma omp parallel
    { (void)omp_get_thread_num(); }
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
    const size_t total_mac = B * matrix_work;
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

    // Determine batch repetition count to control microsecond timing noise
    size_t repetitions = 1;
    if (total_mac < 4096) {
        repetitions = 80;
    } else if (total_mac < 16384) {
        repetitions = 30;
    } else if (total_mac < 65536) {
        repetitions = 10;
    } else if (total_mac < 262144) {
        repetitions = 3;
    } else {
        repetitions = 1;
    }

    struct ModeDef {
        std::string name;
        std::function<void()> fn;
    };

    std::vector<float> out(B * M * N, 0.0f);

    std::vector<ModeDef> modes;
    modes.push_back({"serial_auto", [&]() {
                         for (size_t rep = 0; rep < repetitions; ++rep) {
                             internal::bmm_serial_batches(
                                 a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::Auto
                             );
                         }
                     }});

    if (supports_avx2_backend()) {
        modes.push_back({"serial_avx2", [&]() {
                             for (size_t rep = 0; rep < repetitions; ++rep) {
                                 internal::bmm_serial_batches(
                                     a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::AVX2
                                 );
                             }
                         }});
    }

    modes.push_back({"serial_openmp", [&]() {
                         for (size_t rep = 0; rep < repetitions; ++rep) {
                             internal::bmm_serial_batches(
                                 a.data(), b.data(), out.data(), B, M, K, N, MatMulStrategy::OpenMP
                             );
                         }
                     }});

    modes.push_back({"parallel_batch_serial_inner", [&]() {
                         for (size_t rep = 0; rep < repetitions; ++rep) {
                             internal::bmm_parallel_batches_float(a.data(), b.data(), out.data(), B, M, K, N);
                         }
                     }});

    const double total_operations = operations * static_cast<double>(repetitions);
    double serial_auto_median = 1.0;

    for (size_t i = 0; i < modes.size(); ++i) {
        std::string test_name = modes[i].name + " [B=" + std::to_string(B) + " " + std::to_string(M) + "x" +
                                std::to_string(K) + "x" + std::to_string(N) + "] T=" + std::to_string(threads);

        BenchmarkResult res = BenchmarkRunner::run(test_name, modes[i].fn, iterations, warmups, total_operations);

        const double normalized_median_ms = res.median_ms / static_cast<double>(repetitions);
        const double normalized_average_ms = res.average_ms / static_cast<double>(repetitions);
        const double normalized_stddev_ms = res.stddev_ms / static_cast<double>(repetitions);

        if (modes[i].name == "serial_auto") {
            serial_auto_median = normalized_median_ms;
        }

        const double speedup = (serial_auto_median > 0.0 && normalized_median_ms > 0.0)
                                   ? (serial_auto_median / normalized_median_ms)
                                   : 1.0;

        BmmBenchmarkRecord rec;
        rec.B = B;
        rec.M = M;
        rec.K = K;
        rec.N = N;
        rec.threads = threads;
        rec.mode = modes[i].name;
        rec.median_ms = normalized_median_ms;
        rec.average_ms = normalized_average_ms;
        rec.stddev_ms = normalized_stddev_ms;
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

    enum class BenchmarkSuite { Scout, Refined, LowWork, LargeBoundary };

    BenchmarkSuite suite = BenchmarkSuite::Scout;
    std::string output_path = "output/bmm_benchmark_scout.csv";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--refined") {
            suite = BenchmarkSuite::Refined;
            output_path = "output/bmm_benchmark_refined.csv";
        } else if (arg == "--scout") {
            suite = BenchmarkSuite::Scout;
            output_path = "output/bmm_benchmark_scout.csv";
        } else if (arg == "--low-work") {
            suite = BenchmarkSuite::LowWork;
            output_path = "output/bmm_benchmark_low_work.csv";
        } else if (arg == "--large-boundary") {
            suite = BenchmarkSuite::LargeBoundary;
            output_path = "output/bmm_benchmark_large_boundary.csv";
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
    size_t warmups = 4;
    size_t iterations = 10;

    switch (suite) {
        case BenchmarkSuite::Scout: {
            std::cout << "Running Stage A: Scout Benchmark Suite...\n";
            warmups = 2;
            iterations = 7;

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
            break;
        }

        case BenchmarkSuite::Refined: {
            std::cout << "Running Stage B: Refined Crossover Benchmark Suite...\n";
            warmups = 4;
            iterations = 15;

            for (size_t B : {2, 3, 4, 5, 6, 8, 12, 16}) {
                workloads.push_back({B, 64, 64, 64, "refined_64"});
                workloads.push_back({B, 96, 96, 96, "refined_96"});
                workloads.push_back({B, 128, 128, 128, "refined_128"});
                workloads.push_back({B, 160, 160, 160, "refined_160"});
                workloads.push_back({B, 192, 192, 192, "refined_192"});
            }
            break;
        }

        case BenchmarkSuite::LowWork: {
            std::cout << "Running Stage C: Low-Work Crossover Benchmark Suite...\n";
            warmups = 5;
            iterations = 15;

            // Target MACs: 2048, 4096, 8192, 16384, 32768, 65536, 131072
            // 1. Square small matrices
            for (size_t B : {2, 4, 8, 16}) {
                workloads.push_back({B, 4, 4, 4, "low_work_4"});
                workloads.push_back({B, 8, 8, 8, "low_work_8"});
                workloads.push_back({B, 12, 12, 12, "low_work_12"});
                workloads.push_back({B, 16, 16, 16, "low_work_16"});
                workloads.push_back({B, 20, 20, 20, "low_work_20"});
                workloads.push_back({B, 24, 24, 24, "low_work_24"});
                workloads.push_back({B, 28, 28, 28, "low_work_28"});
                workloads.push_back({B, 32, 32, 32, "low_work_32"});
            }

            // 2. Rectangular small matrices around crossover
            for (size_t B : {2, 4, 8, 16}) {
                workloads.push_back({B, 16, 8, 16, "low_rect_16_8_16"});
                workloads.push_back({B, 32, 16, 32, "low_rect_32_16_32"});
                workloads.push_back({B, 32, 32, 16, "low_rect_32_32_16"});
                workloads.push_back({B, 16, 32, 16, "low_rect_16_32_16"});
            }
            break;
        }

        case BenchmarkSuite::LargeBoundary: {
            std::cout << "Running Stage D: Large-GEMM Boundary Benchmark Suite...\n";
            warmups = 2;
            iterations = 5;

            // Shapes: 256x256x256, 384x384x384, 512x512x512
            // Batches: 2, 4, 6, 8
            for (size_t B : {2, 4, 6, 8}) {
                workloads.push_back({B, 256, 256, 256, "boundary_256"});
                workloads.push_back({B, 384, 384, 384, "boundary_384"});
                workloads.push_back({B, 512, 512, 512, "boundary_512"});
            }
            break;
        }
    }

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
