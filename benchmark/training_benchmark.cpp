#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "benchmark/benchmark_reporter.hpp"
#include "helix.hpp"

using namespace helix;
using namespace helix::benchmark;

void run_linear_regression_benchmark() {
    std::cout << "--- Linear Regression Training ---" << std::endl;
    int num_samples = 1000;
    std::vector<float> X_data(num_samples);
    std::vector<float> Y_data(num_samples);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_x(-5.0f, 5.0f);
    std::normal_distribution<float> dist_noise(0.0f, 0.1f);

    for (int i = 0; i < num_samples; ++i) {
        float x = dist_x(gen);
        X_data[i] = x;
        Y_data[i] = 3.5f * x + 1.2f + dist_noise(gen);
    }

    Tensor X(X_data, {static_cast<size_t>(num_samples), 1});
    Tensor Y(Y_data, {static_cast<size_t>(num_samples), 1});

    Linear model(1, 1);
    SGD optimizer(model.parameters(), 0.01);

    int max_epochs = 1000;
    float tolerance = 0.05f;

    auto start_time = std::chrono::high_resolution_clock::now();

    int converged_epoch = -1;
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        auto pred = model(X);
        auto loss = mse_loss(pred, Y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (loss.item() < tolerance) {
            converged_epoch = epoch;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    if (converged_epoch != -1) {
        std::cout << "Converged in " << converged_epoch << " epochs." << std::endl;
    } else {
        std::cout << "Reached max epochs (" << max_epochs << ") without full convergence." << std::endl;
    }
    std::cout << "Total Training Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average Time per Epoch: "
              << duration.count() / (converged_epoch == -1 ? max_epochs : converged_epoch + 1) << " ms" << std::endl;
    std::cout << std::endl;
}

void run_xor_benchmark() {
    std::cout << "--- XOR MLP Training ---" << std::endl;
    Tensor X({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, Shape{4, 2});
    Tensor Y({0.0f, 1.0f, 1.0f, 0.0f}, Shape{4, 1});

    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));
    SGD optimizer(model.parameters(), 0.1);

    int max_epochs = 5000;
    float tolerance = 0.01f;

    auto start_time = std::chrono::high_resolution_clock::now();

    int converged_epoch = -1;
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        auto pred = model(X);
        auto loss = mse_loss(pred, Y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (loss.item() < tolerance) {
            converged_epoch = epoch;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    if (converged_epoch != -1) {
        std::cout << "Converged in " << converged_epoch << " epochs." << std::endl;
    } else {
        std::cout << "Reached max epochs (" << max_epochs << ") without full convergence." << std::endl;
    }
    std::cout << "Total Training Time: " << duration.count() << " ms" << std::endl;
    std::cout << "Average Time per Epoch: "
              << duration.count() / (converged_epoch == -1 ? max_epochs : converged_epoch + 1) << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    init_autograd();
    BenchmarkReporter::print_header("End-to-End Training Benchmark");

    run_linear_regression_benchmark();
    run_xor_benchmark();

    BenchmarkReporter::print_footer();
    return 0;
}
