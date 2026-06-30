#include "benchmark/benchmark_runner.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "benchmark/timer.hpp"

namespace helix {
    namespace benchmark {

        BenchmarkResult BenchmarkRunner::run(
            const std::string& name,
            std::function<void()> fn,
            size_t iterations,
            size_t warmups,
            double operations_count
        ) {
            if (iterations == 0) {
                throw std::invalid_argument("Benchmark iterations must be greater than 0");
            }

            // Warm-up
            for (size_t i = 0; i < warmups; ++i) {
                fn();
            }

            std::vector<double> times_ms;
            times_ms.reserve(iterations);
            Timer timer;

            for (size_t i = 0; i < iterations; ++i) {
                timer.start();
                fn();
                timer.stop();
                times_ms.push_back(timer.elapsed_ms());
            }

            BenchmarkResult result;
            result.name = name;
            result.iterations = iterations;

            double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
            result.average_ms = sum / iterations;

            auto minmax = std::minmax_element(times_ms.begin(), times_ms.end());
            result.minimum_ms = *minmax.first;
            result.maximum_ms = *minmax.second;

            std::sort(times_ms.begin(), times_ms.end());
            if (iterations % 2 == 0) {
                result.median_ms = (times_ms[iterations / 2 - 1] + times_ms[iterations / 2]) / 2.0;
            } else {
                result.median_ms = times_ms[iterations / 2];
            }

            // Standard deviation
            double variance = 0.0;
            for (double t : times_ms) {
                variance += (t - result.average_ms) * (t - result.average_ms);
            }
            result.stddev_ms = std::sqrt(variance / iterations);

            if (operations_count > 0.0) {
                result.gflops = operations_count / (result.average_ms / 1000.0) / 1e9;
            } else {
                result.gflops = 0.0;
            }

            return result;
        }

    }  // namespace benchmark
}  // namespace helix
