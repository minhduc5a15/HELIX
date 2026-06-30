#pragma once

#include <functional>
#include <string>

#include "benchmark_result.hpp"

namespace helix {
    namespace benchmark {

        class BenchmarkRunner {
        public:
            static BenchmarkResult run(
                const std::string& name,
                std::function<void()> fn,
                size_t iterations = 100,
                size_t warmups = 10,
                double operations_count = 0.0  // To compute GFLOPS
            );
        };

    }  // namespace benchmark
}  // namespace helix
