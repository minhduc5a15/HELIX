#include "benchmark/benchmark_reporter.hpp"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

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

        void BenchmarkReporter::print_comparison(const BenchmarkResult& base, const BenchmarkResult& target) {
            std::cout << "\nComparison (" << base.name << " vs " << target.name << "):\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  " << base.name << ": " << base.average_ms << " ms\n";
            std::cout << "  " << target.name << ": " << target.average_ms << " ms\n";
            
            if (target.average_ms > 0) {
                double speedup = base.average_ms / target.average_ms;
                double improvement = (speedup - 1.0) * 100.0;
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
            std::filesystem::path p(filepath);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }
            std::ofstream out(filepath);
            if (!out.is_open()) return;
            
            out << "Name,Iterations,Average_ms,Minimum_ms,Maximum_ms,Median_ms,StdDev_ms,GFLOPS\n";
            for (const auto& res : results) {
                out << res.name << "," 
                    << res.iterations << "," 
                    << res.average_ms << "," 
                    << res.minimum_ms << "," 
                    << res.maximum_ms << "," 
                    << res.median_ms << "," 
                    << res.stddev_ms << ",";
                if (res.gflops > 0) out << res.gflops;
                out << "\n";
            }
        }

        void BenchmarkReporter::print_footer() {
            std::cout << "========================================================================================\n";
        }

    }  // namespace benchmark
}  // namespace helix
