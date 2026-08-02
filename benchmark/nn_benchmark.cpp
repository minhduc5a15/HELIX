#include <random>
#include <string>

#include "benchmark/benchmark_reporter.hpp"
#include "benchmark/benchmark_runner.hpp"
#include "helix.hpp"

using namespace helix;
using namespace helix::benchmark;

// Builds a one-hot target tensor [batch_size, num_classes], one random class per row.
// CrossEntropyLoss expects a valid target distribution, not arbitrary random values,
// otherwise the measured loss values (though not the timing) are meaningless.
static Tensor make_one_hot_targets(size_t batch_size, size_t num_classes, unsigned seed) {
    std::vector<float> data(batch_size * num_classes, 0.0f);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<size_t> dist_cls(0, num_classes - 1);
    for (size_t i = 0; i < batch_size; ++i) {
        data[i * num_classes + dist_cls(gen)] = 1.0f;
    }
    return Tensor(data, {batch_size, num_classes});
}

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
    // Separate one-hot target for CE: MSE's Y_detached is a regression target
    // (arbitrary real values), CE needs a valid probability distribution per row.
    Tensor Y_onehot_detached = make_one_hot_targets(batch_size, output_dim, /*seed=*/42);

    auto fn_loss = [&]() {
        Tensor l = mse_loss(pred_detached, Y_detached);
        l.backward();  // Include backward of the loss function itself
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "MSELoss (FW+BW)", fn_loss, 30, 5));

    // 3. Forward + Loss + Backward Benchmark (MSE)
    auto fn_fw_bw = [&]() {
        optimizer.zero_grad();
        Tensor p = model(X);
        Tensor l = mse_loss(p, Y);
        l.backward();
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "FW + MSELoss + BW", fn_fw_bw, 30, 5));

    // 2b. CrossEntropyLoss Function Benchmark
    auto fn_ce_loss = [&]() {
        Tensor l = cross_entropy_loss(pred_detached, Y_onehot_detached);
        l.backward();
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "CELoss (FW+BW)", fn_ce_loss, 30, 5));

    // 3b. Forward + CrossEntropyLoss + Backward Benchmark
    Tensor Y_onehot = make_one_hot_targets(batch_size, output_dim, /*seed=*/7);
    auto fn_fw_bw_ce = [&]() {
        optimizer.zero_grad();
        Tensor p = model(X);
        Tensor l = cross_entropy_loss(p, Y_onehot);
        l.backward();
    };
    BenchmarkReporter::print_result(BenchmarkRunner::run(prefix + "FW + CELoss + BW", fn_fw_bw_ce, 30, 5));

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