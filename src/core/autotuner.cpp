#include "core/autotuner.hpp"

#include <chrono>
#include <fstream>
#include <vector>

#include "../backend/matmul/matmul_kernel.hpp"

namespace helix {

    AutoTuner& AutoTuner::get_instance() {
        static AutoTuner instance;
        return instance;
    }

    AutoTuner::AutoTuner() : is_calibrated_(false), omp_threshold_(512ULL * 512ULL * 512ULL) {
        // Initial safe fallback value is the compute volume of 512x512
    }

    std::string AutoTuner::get_cache_filepath() const { return ".helix_autotune"; }

    bool AutoTuner::load_from_cache() {
        std::ifstream file(get_cache_filepath());
        if (file.is_open()) {
            size_t val = 0;
            if (file >> val) {
                omp_threshold_.store(val, std::memory_order_relaxed);
                is_calibrated_.store(true, std::memory_order_release);
                return true;
            }
        }
        return false;
    }

    void AutoTuner::save_to_cache() {
        std::ofstream file(get_cache_filepath());
        if (file.is_open()) {
            file << omp_threshold_.load(std::memory_order_relaxed);
        }
    }

    // Helper function to measure kernel speed
    static double measure_matmul_time(
        const size_t M,
        const size_t K,
        const size_t N,
        void (*func)(const float*, const float*, float*, size_t, size_t, size_t)
    ) {
        const std::vector<float> A(M * K, 1.0f);
        const std::vector<float> B(K * N, 1.0f);
        std::vector<float> C(M * N, 0.0f);

        // Warmup
        func(A.data(), B.data(), C.data(), M, K, N);

        const int iters = (M <= 256) ? 10 : 3;  // Reduce iterations for large matrices to minimize autotune time
        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) {
            func(A.data(), B.data(), C.data(), M, K, N);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count() / iters;
    }

    void AutoTuner::calibrate() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (is_calibrated_.load(std::memory_order_relaxed)) return;

        // Try to load from cache first. If found, skip profiling (ideal for Production restarts).
        if (load_from_cache()) {
            return;
        }

        // Perform Profiling
        // Test 1: Large size (OpenMP might win depending on CPU)
        const double t_avx2_large = measure_matmul_time(512, 512, 512, helix::avx2_micro_matmul);
        const double t_omp_large = measure_matmul_time(512, 512, 512, helix::openmp_matmul);

        // Decision logic
        if (t_avx2_large <= t_omp_large) {
            // AVX2 still wins at 512^3 (possibly due to high OpenMP overhead or low core count)
            // Set threshold to a very high level (1024^3).
            omp_threshold_.store(1024ULL * 1024ULL * 1024ULL, std::memory_order_relaxed);
        } else {
            // At 512, OpenMP wins. Test 256x256 to find the break-even point.
            const double t_avx2_mid = measure_matmul_time(256, 256, 256, helix::avx2_micro_matmul);
            const double t_omp_mid = measure_matmul_time(256, 256, 256, helix::openmp_matmul);

            if (t_omp_mid < t_avx2_mid) {
                omp_threshold_.store(256ULL * 256ULL * 256ULL, std::memory_order_relaxed);
            } else {
                omp_threshold_.store(512ULL * 512ULL * 512ULL, std::memory_order_relaxed);
            }
        }

        is_calibrated_.store(true, std::memory_order_release);
        save_to_cache();
    }

    void AutoTuner::reset_for_testing() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_calibrated_.store(false, std::memory_order_release);
        omp_threshold_.store(512ULL * 512ULL * 512ULL, std::memory_order_relaxed);
    }

    size_t AutoTuner::get_omp_threshold() {
        if (!is_calibrated_.load(std::memory_order_acquire)) {
            calibrate();  // Lazy Evaluation
        }
        return omp_threshold_.load(std::memory_order_relaxed);
    }

    void init_autotuner() { AutoTuner::get_instance().calibrate(); }

}  // namespace helix
