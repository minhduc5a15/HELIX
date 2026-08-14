#include "benchmark/benchmark_reporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace helix::benchmark {

    void BenchmarkReporter::print_header(const std::string& title) {
        std::cout << "========================================================================================\n";
        std::cout << " " << title << "\n";
        std::cout << "========================================================================================\n";
        std::cout << std::left << std::setw(25) << "Name" << std::setw(10) << "Iters" << std::setw(10) << "Avg(ms)"
                  << std::setw(10) << "Min(ms)" << std::setw(10) << "Max(ms)" << std::setw(10) << "Med(ms)"
                  << std::setw(10) << "StdDev" << std::setw(10) << "GFLOPS" << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
    }

    void BenchmarkReporter::print_result(const BenchmarkResult& result) {
        std::cout << std::left << std::setw(25) << result.name << std::setw(10) << result.iterations << std::fixed
                  << std::setprecision(2) << std::setw(10) << result.average_ms << std::setw(10) << result.minimum_ms
                  << std::setw(10) << result.maximum_ms << std::setw(10) << result.median_ms << std::setw(10)
                  << result.stddev_ms;

        if (result.gflops > 0.0) {
            std::cout << std::setw(10) << result.gflops << "\n";
        } else {
            std::cout << std::setw(10) << "N/A" << "\n";
        }
    }

    void BenchmarkReporter::print_comparison(const BenchmarkResult& base, const BenchmarkResult& target) {
        std::cout << "\nComparison (" << base.name << " vs " << target.name << "):\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  " << base.name << ": " << base.average_ms << " ms\n";
        std::cout << "  " << target.name << ": " << target.average_ms << " ms\n";

        if (target.average_ms > 0) {
            const double speedup = base.average_ms / target.average_ms;
            const double improvement = (speedup - 1.0) * 100.0;
            std::cout << "  Speedup: " << speedup << "x\n";
            if (improvement >= 0) {
                std::cout << "  Improvement: +" << improvement << "%\n";
            } else {
                std::cout << "  Improvement: " << improvement << "%\n";
            }
        }
        std::cout << "\n";
    }

    void BenchmarkReporter::export_csv(const std::vector<BenchmarkResult>& results, const std::string& filepath) {
        const std::filesystem::path p(filepath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        std::ofstream out(filepath);
        if (!out.is_open()) return;

        out << "Name,Iterations,Average_ms,Minimum_ms,Maximum_ms,Median_ms,StdDev_ms,GFLOPS\n";
        for (const auto& [name, iterations, average_ms, minimum_ms, maximum_ms, median_ms, stddev_ms, gflops] :
             results) {
            out << name << "," << iterations << "," << average_ms << "," << minimum_ms << "," << maximum_ms << ","
                << median_ms << "," << stddev_ms << ",";
            if (gflops > 0) out << gflops;
            out << "\n";
        }
    }

    void BenchmarkReporter::print_footer() {
        std::cout << "========================================================================================\n";
    }

}  // namespace helix::benchmark