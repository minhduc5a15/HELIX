#include "grad_check.hpp"

#include <cmath>
#include <iostream>

namespace helix {

    bool gradient_check(
        const std::function<Tensor(const std::vector<Tensor>&)>& func,
        const std::vector<Tensor>& inputs,
        const float eps,
        const float tolerance
    ) {
        // 1. Create input tensors with requires_grad = true for autograd
        std::vector<Tensor> inputs_with_grad;
        for (const auto& t : inputs) {
            Tensor cloned = t.clone();
            cloned.set_requires_grad(true);
            inputs_with_grad.push_back(cloned);
        }

        // 2. Run Forward pass
        Tensor out = func(inputs_with_grad);

        if (out.numel() != 1) {
            std::cerr << "[GradientCheck] Error: Function MUST return a scalar tensor." << std::endl;
            return false;
        }

        // 3. Run Backward pass to get gradient
        out.backward();

        // 4. Compute Numerical Gradient using Finite Differences and compare
        bool passed = true;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& orig_t = inputs_with_grad[i];

            // Get AutoGrad for the i-th input and ensure memory is contiguous
            // before accessing via data_ptr()
            Tensor grad = orig_t.grad().contiguous();

            // Loop through each element of the i-th input to compute Numerical Grad
            for (size_t j = 0; j < orig_t.numel(); ++j) {
                // Create an independent copy of inputs for perturbation
                std::vector<Tensor> perturbed_inputs = inputs;
                Tensor p_input = inputs[i].clone();
                p_input.set_requires_grad(false);  // No need to track autograd for numerical pass
                perturbed_inputs[i] = p_input;

                float orig_val = p_input.data_ptr()[j];

                // Compute f(x + eps)
                p_input.data_ptr()[j] = orig_val + eps;
                Tensor out_plus = func(perturbed_inputs);
                float val_plus = out_plus.item();

                // Compute f(x - eps)
                p_input.data_ptr()[j] = orig_val - eps;
                Tensor out_minus = func(perturbed_inputs);
                float val_minus = out_minus.item();

                // Restore original value (not strictly necessary since we copy, but safer)
                p_input.data_ptr()[j] = orig_val;

                // Finite difference formula: (f(x+h) - f(x-h)) / 2h
                float num_grad = (val_plus - val_minus) / (2.0f * eps);
                float auto_grad = grad.data_ptr()[j];

                // If the difference is too large -> fail
                if (std::abs(num_grad - auto_grad) > tolerance) {
                    std::cerr << "[GradientCheck] Mismatch at input " << i << " element " << j << ". Num: " << num_grad
                              << ", Auto: " << auto_grad << ". Diff: " << std::abs(num_grad - auto_grad) << std::endl;
                    passed = false;
                }
            }
        }

        return passed;
    }

}  // namespace helix
