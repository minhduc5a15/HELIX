#include <string>
#include <vector>

#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "helix.hpp"

using namespace helix;
using namespace helix::benchmark;

void run_tensor_ops_benchmark(size_t size) {
    std::string size_str = std::to_string(size) + "x" + std::to_string(size);

    auto fn_create = [size]() { Tensor t(std::vector<float>(size * size, 1.0f), {size, size}); };
    BenchmarkResult res_create = BenchmarkRunner::run("Create " + size_str, fn_create, 30, 5);
    BenchmarkReporter::print_result(res_create);

    Tensor t1(std::vector<float>(size * size, 1.5f), {size, size});
    Tensor t2(std::vector<float>(size * size, 2.0f), {size, size});

    // Element-wise
    auto fn_add = [&]() { Tensor res = t1 + t2; };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Add " + size_str, fn_add, 30, 5));

    auto fn_sub = [&]() { Tensor res = t1 - t2; };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Sub " + size_str, fn_sub, 30, 5));

    auto fn_mul = [&]() { Tensor res = t1 * t2; };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Mul " + size_str, fn_mul, 30, 5));

    auto fn_div = [&]() { Tensor res = t1 / t2; };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Div " + size_str, fn_div, 30, 5));

    // Unary
    auto fn_neg = [&]() { Tensor res = -t1; };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Neg " + size_str, fn_neg, 30, 5));

    auto fn_relu = [&]() { Tensor res = t1.relu(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("ReLU " + size_str, fn_relu, 30, 5));

    // Reduction
    auto fn_sum = [&]() { Tensor res = t1.sum(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Sum " + size_str, fn_sum, 30, 5));

    auto fn_mean = [&]() { Tensor res = t1.mean(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Mean " + size_str, fn_mean, 30, 5));

    // View Operations (Contiguous vs Non-contiguous)
    auto fn_reshape = [&]() { Tensor res = t1.reshape({size * size}); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Reshape (Contig) " + size_str, fn_reshape, 30, 5));

    auto fn_transpose = [&]() { Tensor res = t1.transpose(0, 1); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Transpose (Contig) " + size_str, fn_transpose, 30, 5));

    auto fn_flatten = [&]() { Tensor res = t1.flatten(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Flatten (Contig) " + size_str, fn_flatten, 30, 5));

    // Non-contiguous views (will trigger copy in reshape/flatten)
    Tensor t_nc = t1.transpose(0, 1); 

    auto fn_reshape_nc = [&]() { Tensor res = t_nc.reshape({size * size}); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Reshape (Non-Contig) " + size_str, fn_reshape_nc, 30, 5));

    auto fn_flatten_nc = [&]() { Tensor res = t_nc.flatten(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run("Flatten (Non-Contig) " + size_str, fn_flatten_nc, 30, 5));
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
