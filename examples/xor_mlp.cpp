#include <iomanip>
#include <iostream>
#include <vector>

#include "helix.hpp"

using namespace helix;

int main() {
    init_autograd();
    std::cout << "============================================" << std::endl;
    std::cout << "          HELIX XOR MLP DEMO" << std::endl;
    std::cout << "============================================" << std::endl;

    // XOR Dataset
    // Inputs: (0,0), (0,1), (1,0), (1,1)
    // Outputs: 0, 1, 1, 0
    std::vector<float> X_data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    std::vector<float> Y_data = {0.0f, 1.0f, 1.0f, 0.0f};

    auto X = Tensor(X_data, {4, 2});
    auto Y = Tensor(Y_data, {4, 1});  // target

    // Define MLP model: 2 -> 4 -> 1
    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));

    // Define Optimizer: SGD with learning rate 0.1
    SGD optimizer(model.parameters(), 0.1);

    // Training loop
    int epochs = 1000;
    std::cout << "Training for " << epochs << " epochs..." << std::endl;
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

        if (epoch % 100 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " - Loss: " << loss.item() << std::endl;
        }
    }

    // Evaluation
    std::cout << "============================================" << std::endl;
    std::cout << "Evaluation (Predictions vs Targets):" << std::endl;
    auto pred = model.forward(X);
    const float* pred_data = pred.data_ptr();
    const float* target_data = Y.data_ptr();

    bool all_correct = true;
    for (int i = 0; i < 4; ++i) {
        float p = pred_data[i];
        float t = target_data[i];
        std::cout << "Input: (" << X_data[i * 2] << ", " << X_data[i * 2 + 1] << ") " << "-> Target: " << t
                  << " | Predicted: " << std::fixed << std::setprecision(4) << p << std::endl;

        // Check if prediction is correct (threshold 0.5)
        if ((t == 1.0f && p < 0.5f) || (t == 0.0f && p >= 0.5f)) {
            all_correct = false;
        }
    }

    std::cout << "============================================" << std::endl;
    if (all_correct) {
        std::cout << "Result: SUCCESS! The MLP successfully learned the XOR function." << std::endl;
    } else {
        std::cout << "Result: FAILED! The model did not converge on the XOR function." << std::endl;
    }

    return 0;
}
