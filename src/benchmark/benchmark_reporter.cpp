#include "benchmark/benchmark_reporter.hpp"

#include <iomanip>
#include <iostream>

namespace helix {
    namespace benchmark {

        void BenchmarkReporter::print_header(const std::string& title) {
            std::cout << "========================================================================================\n";
            std::cout << " " << title << "\n";
            std::cout << "========================================================================================\n";
            std::cout << std::left << std::setw(25) << "Name" << std::setw(10) << "Iters" << std::setw(10) << "Avg(ms)"
                      << std::setw(10) << "Min(ms)" << std::setw(10) << "Max(ms)" << std::setw(10) << "Med(ms)"
                      << std::setw(10) << "StdDev" << std::setw(10) << "GFLOPS" << "\n";
            std::cout << "----------------------------------------------------------------------------------------\n";
        }

        void BenchmarkReporter::print_result(const BenchmarkResult& res) {
            std::cout << std::left << std::setw(25) << res.name << std::setw(10) << res.iterations << std::fixed
                      << std::setprecision(2) << std::setw(10) << res.average_ms << std::setw(10) << res.minimum_ms
                      << std::setw(10) << res.maximum_ms << std::setw(10) << res.median_ms << std::setw(10)
                      << res.stddev_ms;

            if (res.gflops > 0.0) {
                std::cout << std::setw(10) << res.gflops << "\n";
            } else {
                std::cout << std::setw(10) << "N/A" << "\n";
            }
        }

        void BenchmarkReporter::print_footer() {
            std::cout << "========================================================================================\n";
        }

    }  // namespace benchmark
}  // namespace helix
