#pragma once

#include <cstddef>
#include <string>

namespace helix {
    namespace benchmark {

        struct BenchmarkResult {
            std::string name;
            size_t iterations;

            double average_ms;
            double minimum_ms;
            double maximum_ms;
            double median_ms;
            double stddev_ms;

            double gflops;
        };

    }  // namespace benchmark
}  // namespace helix
