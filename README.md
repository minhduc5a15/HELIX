# HELIX: A Modern C++ Autograd & Neural Network Framework

![HELIX Logo](https://img.shields.io/badge/C%2B%2B-20-blue.svg) ![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg)

**HELIX** is a lightweight, high-performance deep learning framework built entirely from scratch in C++20. Designed for educational purposes, rapid prototyping, and high-performance execution on CPU architectures, HELIX provides a modern Autograd engine, a rich set of Neural Network modules, and highly optimized backend execution paths leveraging SIMD (AVX2/FMA) and OpenMP multi-threading.

## Key Features

- **Dynamic Computational Graph (Autograd)**: Define-by-run automatic differentiation similar to PyTorch, supporting complex forward and backward passes.
- **High-Performance CPU Backend**:
  - Cache-oblivious and Tiled matrix multiplication algorithms for maximum L1/L2 cache utilization.
  - Runtime detection and dispatch for **AVX2 & FMA** SIMD instructions, processing 8 floating-point operations per cycle.
  - Multi-threading support powered by **OpenMP** for parallel execution on multi-core CPUs.
- **Extensible Neural Network API**: Built-in support for `Linear`, `ReLU`, `Sequential`, and MSE Loss, designed with an intuitive API inspired by PyTorch (`helix::nn`).
- **Comprehensive Tensor Operations**: Multidimensional `Tensor` class supporting automatic broadcasting, manipulation (reshape, view, transpose), and reduction operations.
- **Memory Safety & Modern C++**: Built using strictly modern C++20 standard practices (smart pointers, templates, `std::variant`, auto) ensuring a robust, leak-free memory architecture.

## Getting Started

### Prerequisites

- CMake 3.25+
- A C++20 compatible compiler (GCC 11+, Clang 14+, or MSVC 19.30+)
- OpenMP support (optional, but recommended for maximum performance)

### Building the Project

```bash
# Clone the repository
git clone https://github.com/your-org/HELIX.git
cd HELIX

# Build using the provided build script
./build.sh
```

### Running Tests and Benchmarks

HELIX features an extensive test suite and benchmark programs.

```bash
# Run all tests
./run_tests.sh

# Run Matrix Multiplication Benchmark
./out/build/HELIX/benchmark/matmul_benchmark
```

## Quick Start Example

Here is a quick example of defining and training a neural network in HELIX to learn the XOR function:

```cpp
#include <helix/helix.hpp>
#include <iostream>

using namespace helix;

int main() {
    // Inputs: (0,0), (0,1), (1,0), (1,1) -> Targets: 0, 1, 1, 0
    auto X = tensor({0.f, 0.f, 0.f, 1.f, 1.f, 0.f, 1.f, 1.f}, {4, 2});
    auto Y = tensor({0.f, 1.f, 1.f, 0.f}, {4, 1});

    // Define MLP: 2 -> 4 -> 1
    Sequential model(
        std::make_shared<Linear>(2, 4),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(4, 1)
    );

    // Stochastic Gradient Descent
    SGD optimizer(model.parameters(), 0.1);

    for (int epoch = 0; epoch < 1000; ++epoch) {
        auto pred = model.forward(X);
        auto loss = mse_loss(pred, Y);
        
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();
    }
    
    std::cout << "Training complete!" << std::endl;
    return 0;
}
```

Check out the `examples/` directory for more demos, including Linear Regression and XOR MLP training!

## Architecture Details

HELIX is composed of the following key layers:
1. **Core**: Multidimensional Tensors, Dispatcher, Allocators, and Shape Broadcast semantics.
2. **Backend**: Optimized compute kernels (`matmul_naive`, `matmul_tiled`, `matmul_avx2`, `reduce`, `ops`).
3. **Autograd**: DAG construction, `BackwardEngine`, gradient accumulation.
4. **NN Modules**: High-level layer abstractions, parameter management, Optimizers (SGD), and Loss functions.

## License

This project is licensed under the MIT License.
