#pragma once

#include "benchmark_result.hpp"

namespace helix {
    namespace benchmark {

        class BenchmarkReporter {
        public:
            static void print_header(const std::string& title);
            static void print_result(const BenchmarkResult& result);
            static void print_footer();
        };

    }  // namespace benchmark
}  // namespace helix
