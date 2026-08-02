# Developer Guide

This document is intended for developers who wish to extend HELIX by adding new Layers, Optimizers, Backends, or Tensor Operations.

---

## 1. Adding a New Neural Network Layer

All Layers in HELIX inherit from the base class `helix::nn::Module`.
To add a new Layer (e.g., `Sigmoid` or `Conv2d`), you need to:

1. **Inherit from Module**: Create a new class inheriting from `helix::nn::Module`.
2. **Declare Weights (if any)**: Register weight Tensors (e.g., `weight`, `bias`) using the `register_parameter()` method. This allows the system to automatically recognize parameters to push to the Optimizer.
3. **Override the `forward()` method**: Define the computation logic for the Forward Pass. Autograd will automatically handle the Backward pass.

**Example of a Custom Layer (Without weights):**

```cpp
namespace helix::nn {
class Sigmoid : public Module {
public:
    std::shared_ptr<core::Tensor> forward(std::shared_ptr<core::Tensor> input) override {
        // Reuse Tensor operations to implement Sigmoid: 1 / (1 + exp(-x))
        auto neg_x = core::Tensor::neg(input);
        auto exp_x = core::Tensor::exp(neg_x);
        auto one = std::make_shared<core::Tensor>(1.0f);
        auto sum = core::Tensor::add(one, exp_x);
        return core::Tensor::div(one, sum);
    }
};
}
```

---

## 2. Adding a New Loss Function

By design, Loss functions **DO NOT** inherit from `Module`. This is because `Module::forward()` is strictly designed for single-input operations, whereas Loss functions inherently require two inputs (`predictions` and `targets`).
Instead, Loss functions are implemented as **Free Functions**.

1. **Declare the Function**: Add the declaration in `include/nn/loss.hpp`. Example: `Tensor mse_loss(const Tensor& pred, const Tensor& target);`
2. **Define the Computation**: In `src/nn/loss.cpp`, define the computation graph.
3. **Autograd Hooking**: Create a class inheriting from `autograd::Node` for the backward pass of the loss (e.g., `MSELossBackward`), and link it to the resulting scalar `Tensor` via `set_grad_fn()`.

**Example (MSE Loss):**

```cpp
Tensor mse_loss(const Tensor& pred, const Tensor& target) {
    auto diff = Tensor::sub(pred, target);
    auto sq = Tensor::pow(diff, 2.0f);
    auto sum = Tensor::sum(sq);
    auto mean = Tensor::div_scalar(sum, static_cast<float>(pred.size()));
    
    // Attach custom backward node to `mean` Tensor here...
    return mean;
}
```

---

## 3. Adding a Tensor Operation

Adding an operation directly on a Tensor is more complex because it requires interacting with both Autograd and the Dispatcher.

**Step 1: Backend Kernel**
Create an execution function (running a for loop) in `src/backend/kernels/ops.cpp`.

**Step 2: Dispatcher**
Add a caller function to the `Dispatcher` (`src/core/dispatcher.cpp`). Here you define which Backend configuration will handle this task.

**Step 3: Autograd Node**
Create a class inheriting from `autograd::Node` (in `src/autograd/function.cpp`).
Override the `apply()` method to compute the derivative (Backward) for your operation.

**Step 4: Attach to Tensor API**
Add an interface function to `include/core/tensor.hpp`.
In this function:

- Create an `output` Tensor.
- Call the `Dispatcher` to run the Kernel.
- Generate an Autograd Node (if `requires_grad`) and establish a parent-child link (`grad_fn`).

---

## 4. Adding a New Backend (e.g., Vulkan / CUDA)
>
> **Note:** Backends like CUDA, Vulkan, or Metal are only used to illustrate how to extend HELIX's architecture in the future. The current version of the framework has not implemented these backends.

HELIX is designed with a clear separation between Backend and Dispatcher.

1. Create a separate implementation directory/files (e.g., `src/backend/cuda/`).
2. Write Kernel sets (MatMul, Ops, Reduce) utilizing the technology of that Backend.
3. Open `src/core/dispatcher.cpp` and add priority logic to select the new Backend if the hardware device supports it.
4. Ensure the CMake structure (`CMakeLists.txt`) only compiles that Backend when the corresponding system flag is enabled (e.g., `-DUSE_CUDA=ON`).
