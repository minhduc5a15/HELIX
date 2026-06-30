#pragma once

#include "benchmark_result.hpp"
#include <vector>

namespace helix {
    namespace benchmark {

        class BenchmarkReporter {
        public:
            static void print_header(const std::string& title);
            static void print_result(const BenchmarkResult& result);
            static void print_comparison(const BenchmarkResult& base, const BenchmarkResult& target);
            static void export_csv(const std::vector<BenchmarkResult>& results, const std::string& filepath);
            static void print_footer();
        };

    }  // namespace benchmark
}  // namespace helix
