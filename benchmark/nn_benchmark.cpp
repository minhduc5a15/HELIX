#include <string>
#include <vector>

#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "helix.hpp"

using namespace helix;
using namespace helix::benchmark;

void run_nn_benchmark(size_t batch_size, size_t input_dim, size_t hidden_dim, size_t output_dim) {
    std::string prefix = "Batch " + std::to_string(batch_size) + " ";
    
    // Create dummy data
    Tensor X = Tensor::randn({batch_size, input_dim});
    Tensor Y = Tensor::randn({batch_size, output_dim});

    Sequential model(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, output_dim));
    SGD optimizer(model.parameters(), 0.01f);

    // 1. Forward Benchmark
    auto fn_forward = [&]() { Tensor pred = model(X); };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "Forward", fn_forward, 30, 5));

    // 2. Loss Function Benchmark (using detached tensors to isolate loss overhead)
    Tensor pred_detached = Tensor::randn({batch_size, output_dim});
    pred_detached.set_requires_grad(true);
    Tensor Y_detached = Tensor::randn({batch_size, output_dim});
    
    auto fn_loss = [&]() { 
        Tensor l = mse_loss(pred_detached, Y_detached); 
        l.backward(); // Include backward of the loss function itself
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "MSELoss (FW+BW)", fn_loss, 30, 5));

    // 3. Forward + Loss + Backward Benchmark
    auto fn_fw_bw = [&]() { 
        optimizer.zero_grad();
        Tensor p = model(X);
        Tensor l = mse_loss(p, Y);
        l.backward(); 
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "FW + Loss + BW", fn_fw_bw, 30, 5));

    // 4. Optimizer Benchmark
    auto fn_optim = [&]() { optimizer.step(); };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "Optimizer.step()", fn_optim, 30, 5));

    // 5. Full Epoch Benchmark
    auto fn_epoch = [&]() {
        optimizer.zero_grad();
        Tensor p = model(X);
        Tensor l = mse_loss(p, Y);
        l.backward();
        optimizer.step();
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "Full Epoch", fn_epoch, 30, 5));
}

int main() {
    init_autograd();
    BenchmarkReporter::print_header("Neural Network Benchmark");

    // Small Batch
    run_nn_benchmark(32, 128, 256, 10);
    // Large Batch
    run_nn_benchmark(256, 128, 256, 10);

    BenchmarkReporter::print_footer();
    return 0;
}
