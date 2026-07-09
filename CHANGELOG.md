# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-10

### Added
- **Tensor Runtime**:
  - N-dimensional tensor support with customizable shapes, strides, and memory storage.
  - Zero-memory broadcasting across multiple dimensions.
  - O(1) view operations (`reshape`, `transpose`, `flatten`, `slice`).
  - Element-wise operations, reductions (`sum`, `mean`), and scalar math.
- **Autograd Engine**:
  - Dynamic computational graph (Define-by-Run) generation.
  - Reverse-mode Automatic Differentiation.
  - Efficient gradient accumulation and graph memory management via smart pointers.
- **Neural Network Core**:
  - `Module` base class for composable neural network components.
  - Core layers: `Linear`, `Sequential`.
  - Activations: `ReLU`, `Sigmoid`.
  - Loss Functions: `MSELoss`.
- **Optimizers**:
  - Stochastic Gradient Descent (`SGD`) optimizer.
- **High-Performance MatMul Backends**:
  - **Naive**: Absolute baseline reference for correctness.
  - **Blocked**: Cache tiling optimization for improved cache hit rates.
  - **SIMD AVX2**: Register-level hardware instruction optimization.
  - **OpenMP**: Multi-threading optimization for large matrix dimensions.
- **Testing & Benchmarking**:
  - Comprehensive Unit Test suite via GoogleTest covering all ops and backward passes.
  - Automated Benchmarking infrastructure and reports comparing GFLOPS performance across backends.
- **Documentation**:
  - Architecture Guide, Design Decisions, and Developer Guide.
  - Doxygen integration.

### Changed
- Extensive code quality and static analysis refactoring to enforce strict compilation warnings and performance safety.
- Refactored `Tensor` class for zero-overhead move semantics.
