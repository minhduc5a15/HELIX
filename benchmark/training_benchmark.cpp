#include <chrono>
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

    // --- Classification Training Benchmark ---
    std::cout << "--- Classification Training Benchmark ---" << std::endl;
    int num_samples = 1000;
    int input_dim = 10;
    int num_classes = 5;

    Tensor X_cls = Tensor::randn({static_cast<size_t>(num_samples), static_cast<size_t>(input_dim)});
    // One-hot targets
    std::vector<float> y_cls_data(num_samples * num_classes, 0.0f);
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist_cls(0, num_classes - 1);
    for (int i = 0; i < num_samples; ++i) {
        y_cls_data[i * num_classes + dist_cls(gen)] = 1.0f;
    }
    Tensor Y_cls(y_cls_data, {static_cast<size_t>(num_samples), static_cast<size_t>(num_classes)});

    Sequential cls_model(Linear(input_dim, 32), ReLU(), Linear(32, num_classes));
    SGD cls_optimizer(cls_model.parameters(), 0.05);

    int max_cls_epochs = 100;
    auto start_time_cls = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < max_cls_epochs; ++epoch) {
        auto pred = cls_model(X_cls);
        auto loss = cross_entropy_loss(pred, Y_cls);

        cls_optimizer.zero_grad();
        loss.backward();
        cls_optimizer.step();
    }

    auto end_time_cls = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_cls = end_time_cls - start_time_cls;

    std::cout << "Ran " << max_cls_epochs << " epochs on Classification task." << std::endl;
    std::cout << "Total Training Time: " << duration_cls.count() << " ms" << std::endl;
    std::cout << "Average Time per Epoch: " << duration_cls.count() / max_cls_epochs << " ms" << std::endl;
    std::cout << std::endl;

    BenchmarkReporter::print_footer();
    return 0;
}
