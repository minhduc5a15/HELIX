# HELIX

HELIX is a small deep learning framework implemented in C++20. It is intended for studying tensor runtimes, reverse-mode automatic differentiation, CPU kernels, and neural-network training without depending on an existing tensor library.

The current implementation is CPU-only. It can train multilayer perceptrons and includes an MNIST example, but it is not a replacement for a production framework such as PyTorch or TensorFlow.

## Implemented features

- N-dimensional tensors with shapes, strides, broadcasting, slicing, transposition, reshaping, cloning, and concatenation (`helix::cat`).
- `Float32`, `Float64`, `Int32`, and `Int64` dtypes with type promotion.
- Reductions including sum, mean, and index reduction (`Tensor::argmax`).
- Dynamic reverse-mode autograd for tensor arithmetic, reductions, matrix multiplication, supported view operations, concatenation, and the provided losses.
- RAII-based autograd control (`helix::no_grad`) with thread-local graph suppression for zero-overhead evaluation and inference.
- `Linear`, `ReLU`, and `Sequential` neural-network components.
- Mean squared error and numerically stable cross entropy with one-hot targets.
- SGD and gradient clearing.
- CPU matrix multiplication backends: naive, blocked, AVX2/FMA, and OpenMP.
- Runtime calibration for choosing between the single-threaded AVX2 and OpenMP float matrix-multiplication paths.
- Unit, gradient, integration, and stress tests, plus standalone benchmark programs.

## Current scope and limitations

- Computation is implemented only for `DeviceType::CPU`. `DeviceType::CUDA` is declared as API metadata, but no CUDA allocator or kernels exist.
- `matmul` accepts two 2D tensors; batched matrix multiplication is not implemented.
- `cross_entropy_loss` expects predictions and one-hot targets with the same `[batch, classes]` shape.
- Integer tensors cannot require gradients.
- Autograd must be initialized by calling `init_autograd()` before creating tensors or modules that require gradients.
- In-place addition on tensors tracked by autograd is rejected. In-place operations on internally overlapping views are also rejected.
- Model serialization, convolution, normalization, dropout, embeddings, and GPU execution are not implemented.

See [Architecture](docs/architecture.md) for the runtime design and [Developer Guide](docs/developer_guide.md) for extension points.

## Requirements

- CMake 3.25 or newer.
- A C++20 compiler.
- Ninja for the checked-in CMake preset.
- OpenMP is optional. CMake uses it when a compatible implementation is found.
- On x86-64, the default build currently compiles with AVX2 and FMA flags. The resulting binary therefore requires a compatible CPU.

On Ubuntu, the usual development packages can be installed with:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build
```

Install Clang and `libomp-dev` as well when using the ThreadSanitizer configuration.

## Build and test

```bash
git clone https://github.com/minhduc5a15/HELIX.git
cd HELIX

./build.sh --release
./run_tests.sh
```

`build.sh` configures and builds into `build/`. Useful variants are:

```bash
./build.sh --clean --debug
./build.sh --clean --release
./build.sh --clean --tsan
```

The TSan option requires Clang. The test script forwards additional arguments to CTest, for example:

```bash
./run_tests.sh -R Autograd
```

## Minimal training example

```cpp
#include <vector>

#include "helix.hpp"

using namespace helix;

int main() {
    init_autograd();

    const Tensor inputs(
        std::vector<float>{0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 1.0F},
        Shape{4, 2}
    );
    const Tensor targets(std::vector<float>{0.0F, 1.0F, 1.0F, 0.0F}, Shape{4, 1});

    Sequential model(Linear(2, 4), ReLU(), Linear(4, 1));
    SGD optimizer(model.parameters(), 0.1F);

    for (int epoch = 0; epoch < 1000; ++epoch) {
        optimizer.zero_grad();
        Tensor prediction = model(inputs);
        Tensor loss = mse_loss(prediction, targets);
        loss.backward();
        optimizer.step();
    }

    // Evaluation without autograd overhead
    {
        no_grad guard;
        Tensor prediction = model(inputs);
        Tensor classes = prediction.argmax(1);
    }
}
```

The repository builds equivalent runnable examples under `build/examples/`:

```bash
./build/examples/xor_mlp
./build/examples/linear_regression
./build/examples/logic_gates
```

## MNIST example

The MNIST example trains a `784 -> 256 -> 128 -> 10` multilayer perceptron with ReLU activations and one-hot cross entropy.

```bash
bash scripts/download_mnist.sh
./build.sh --release
./build/examples/mnist
```

The example reads files from `data/mnist/`, trains for ten epochs with batches of 64, and reports measured loss, accuracy, and epoch time. Results depend on the compiler, CPU, OpenMP runtime, and random initialization.

## Benchmarks

Build and execute all benchmark programs with:

```bash
./build.sh --release
./run_benchmark.sh
```

The matrix-multiplication benchmark writes `output/matmul_benchmark.csv`. Benchmark results are machine-specific; the repository does not treat a single recorded GFLOPS value as a portable performance guarantee. See [Benchmark Guide](docs/benchmark_report.md) for the measurement method and interpretation notes.

## Documentation

- [Architecture](docs/architecture.md)
- [Design Decisions](docs/design_decisions.md)
- [Developer Guide](docs/developer_guide.md)
- [Benchmark Guide](docs/benchmark_report.md)
- [Coding Convention](docs/coding_convention.md)

The public include entry point is [`include/helix.hpp`](include/helix.hpp). All user-facing tensor, neural-network, loss, and optimizer types currently live directly in the `helix` namespace.
