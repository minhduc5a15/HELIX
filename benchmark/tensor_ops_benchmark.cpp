#include <string>

#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "helix.hpp"

using namespace helix;
using namespace helix::benchmark;

void run_tensor_ops_benchmark(size_t size) {
    std::string size_str = std::to_string(size) + "x" + std::to_string(size);

    auto fn_create = [size]() { Tensor t(std::vector<float>(size * size, 1.0f), {size, size}); };
    BenchmarkResult res_create = BenchmarkRunner::run("Create " + size_str, fn_create, 50, 5);
    BenchmarkReporter::print_result(res_create);

    Tensor t1(std::vector<float>(size * size, 1.0f), {size, size});
    Tensor t2(std::vector<float>(size * size, 2.0f), {size, size});

    auto fn_add = [&]() { Tensor res = t1 + t2; };
    BenchmarkResult res_add = BenchmarkRunner::run("Add " + size_str, fn_add, 50, 5);
    BenchmarkReporter::print_result(res_add);

    auto fn_mean = [&]() { Tensor res = t1.mean(); };
    BenchmarkResult res_mean = BenchmarkRunner::run("Mean " + size_str, fn_mean, 50, 5);
    BenchmarkReporter::print_result(res_mean);
}

int main() {
    BenchmarkReporter::print_header("Tensor Ops Benchmark");

    std::vector<size_t> sizes = {128, 512, 1024};
    for (size_t size : sizes) {
        run_tensor_ops_benchmark(size);
    }

    BenchmarkReporter::print_footer();
    return 0;
}
