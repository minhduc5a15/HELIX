#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "helix.hpp"

using namespace helix;

int main() {
    init_autograd();
    std::cout << "============================================" << std::endl;
    std::cout << "       HELIX LINEAR REGRESSION DEMO" << std::endl;
    std::cout << "============================================" << std::endl;

    // Generate synthetic dataset for y = 3.5 * x + 1.2
    int num_samples = 100;
    std::vector<float> X_data(num_samples);
    std::vector<float> Y_data(num_samples);

    std::mt19937 gen(42);  // fixed seed
    std::uniform_real_distribution<float> dist_x(-5.0f, 5.0f);
    std::normal_distribution<float> dist_noise(0.0f, 0.1f);

    for (int i = 0; i < num_samples; ++i) {
        float x = dist_x(gen);
        float noise = dist_noise(gen);
        X_data[i] = x;
        Y_data[i] = 3.5f * x + 1.2f + noise;
    }

    auto X = Tensor(X_data, {static_cast<size_t>(num_samples), 1});
    auto Y = Tensor(Y_data, {static_cast<size_t>(num_samples), 1});

    // Define model: y = w * x + b
    Linear model(1, 1);

    // Initial parameters
    auto params = model.named_parameters();
    Tensor weight = params[0].second;
    Tensor bias = params[1].second;

    std::cout << "Initial Weights: " << weight.data_ptr()[0] << ", Initial Bias: " << bias.data_ptr()[0] << std::endl;

    // Define Optimizer: SGD with learning rate 0.01
    SGD optimizer(model.parameters(), 0.01);

    // Training loop
    int epochs = 500;
    std::cout << "\nTraining for " << epochs << " epochs..." << std::endl;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto pred = model.forward(X);

        // Compute loss (MSE)
        auto loss = mse_loss(pred, Y);

        // Zero gradients
        optimizer.zero_grad();

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer.step();

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " - Loss: " << std::fixed << std::setprecision(6) << loss.item()
                      << " | w: " << weight.data_ptr()[0] << ", b: " << bias.data_ptr()[0] << std::endl;
        }
    }

    // Evaluation
    std::cout << "============================================" << std::endl;
    std::cout << "Final Parameters:" << std::endl;
    float final_w = weight.data_ptr()[0];
    float final_b = bias.data_ptr()[0];
    std::cout << "Estimated w: " << final_w << " (Expected ~3.5)" << std::endl;
    std::cout << "Estimated b: " << final_b << " (Expected ~1.2)" << std::endl;

    if (std::abs(final_w - 3.5f) < 0.1f && std::abs(final_b - 1.2f) < 0.1f) {
        std::cout << "Result: SUCCESS! The model accurately approximated the linear function." << std::endl;
    } else {
        std::cout << "Result: FAILED! The model parameters are far from expected values." << std::endl;
    }

    return 0;
}
