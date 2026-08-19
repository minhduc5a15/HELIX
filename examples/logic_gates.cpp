#include <iomanip>
#include <iostream>
#include <vector>

#include "helix.hpp"

using namespace helix;

void train_and_eval_gate(const std::string& gate_name, const std::vector<float>& Y_data) {
    std::cout << "\n============================================" << std::endl;
    std::cout << "          HELIX " << gate_name << " GATE DEMO" << std::endl;
    std::cout << "============================================" << std::endl;

    std::vector<float> X_data = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};

    auto X = Tensor(X_data, {4, 2});
    auto Y = Tensor(Y_data, {4, 1});

    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));
    SGD optimizer(model.parameters(), 0.1);

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto pred = model.forward(X);
        auto loss = mse_loss(pred, Y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    auto pred = model.forward(X);
    const float* pred_data = pred.data_ptr();
    const float* target_data = Y.data_ptr();

    bool all_correct = true;
    for (int i = 0; i < 4; ++i) {
        float p = pred_data[i];
        float t = target_data[i];
        std::cout << "Input: (" << X_data[i * 2] << ", " << X_data[i * 2 + 1] << ") " << "-> Target: " << t
                  << " | Predicted: " << std::fixed << std::setprecision(4) << p << std::endl;

        if ((t == 1.0f && p < 0.5f) || (t == 0.0f && p >= 0.5f)) {
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "Result: SUCCESS! The model successfully learned the " << gate_name << " function." << std::endl;
    } else {
        std::cout << "Result: FAILED! The model did not converge on the " << gate_name << " function." << std::endl;
    }
}

int main() {
    init_autograd();

    // AND Gate: 0, 0, 0, 1
    train_and_eval_gate("AND", {0.0f, 0.0f, 0.0f, 1.0f});

    // OR Gate: 0, 1, 1, 1
    train_and_eval_gate("OR", {0.0f, 1.0f, 1.0f, 1.0f});

    // NAND Gate: 1, 1, 1, 0
    train_and_eval_gate("NAND", {1.0f, 1.0f, 1.0f, 0.0f});

    return 0;
}
