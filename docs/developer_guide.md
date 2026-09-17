# Developer Guide

This guide describes the extension mechanisms present in the current codebase. Public tensor, neural-network, loss, and optimizer APIs live directly in `namespace helix`. Benchmark helpers use `namespace helix::benchmark`.

## Build and verification

Configure, build, and run the complete registered test suite with:

```bash
./build.sh --debug
./run_tests.sh
```

For a clean optimized build:

```bash
./build.sh --clean --release
```

The root CMake project builds the static `helix` library, tests, stress tests, benchmarks, and examples. New implementation files must also be added to the `helix` source list in the root `CMakeLists.txt`.

Autograd tests and programs must call `init_autograd()` before setting `requires_grad` or constructing parameterized modules.

## Adding a module

Derive from `helix::Module` and implement:

```cpp
Tensor forward(const Tensor& input) override;
```

A stateless module can be composed entirely from existing Tensor operations:

```cpp
#include "helix.hpp"

namespace helix {

class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return input.tanh();
    }
};

}  // namespace helix
```

Autograd records the operations used inside `forward`; a separate module-specific backward class is unnecessary when existing differentiable Tensor operations are sufficient.

A module with trainable tensors must store those tensors, call `set_requires_grad(true)` during construction, and override `named_parameters()`:

```cpp
std::vector<std::pair<std::string, Tensor>> named_parameters() override {
    return {{"weight", weight_}, {"bias", bias_}};
}
```

`Module::parameters()` derives its result from `named_parameters()`. There is no `register_parameter()` API in the current implementation.

`Sequential` accepts either a vector of `shared_ptr<Module>` or concrete module objects through its variadic constructor. Parameter names are prefixed with their layer index, such as `0.weight`.

## Adding a loss

Losses are free functions declared in `include/nn/loss.hpp` and defined in `src/nn/loss.cpp`.

When a loss can be expressed with existing differentiable Tensor operations, compose those operations and allow the graph builder to record them. The current mean squared error follows this pattern:

```cpp
Tensor mse_loss(const Tensor& prediction, const Tensor& target) {
    const Tensor difference = prediction - target;
    return (difference * difference).mean();
}
```

For a fused loss kernel, follow the cross-entropy implementation:

1. declare the public free function;
2. add a Dispatcher entry and backend kernel;
3. add an `OpType` and populate its `OperationContext`;
4. implement a `Node::backward` subclass;
5. construct that node in `AutogradGraphBuilder`;
6. test the value, invalid inputs, non-contiguous inputs where supported, and numerical gradients.

The graph builder links one edge per forward input. A backward node must therefore return the same number of gradient entries as it has edges. Use an empty `Tensor` for an input whose edge exists but whose gradient is intentionally unused, as cross entropy does for its target.

## Adding a Tensor operation

A differentiable Tensor operation normally touches these locations:

1. public declaration and forwarding method in `include/core/tensor.hpp`;
2. backend declaration in `include/backend/cpu_backend.hpp` when a separate kernel is needed;
3. kernel implementation under `src/backend/`;
4. Dispatcher declaration and implementation;
5. `OpType` in `include/core/graph_builder.hpp`;
6. backward node declaration in `include/autograd/function.hpp` and definition in `src/autograd/function.cpp`;
7. node creation in `src/autograd/graph_builder.cpp`;
8. unit tests and, for differentiable floating-point operations, a numerical gradient check.

Dispatcher implementations must preserve and validate the following properties where relevant:

- shape and broadcasting rules;
- dtype promotion;
- device agreement;
- contiguous versus strided access;
- aliasing and internal overlap for mutation;
- graph construction using the original logical inputs rather than temporary broadcast or cast tensors.

Backward implementations should use `SavedTensor` for forward values needed during differentiation. It detects later in-place changes through the shared storage version counter and avoids retaining the forward autograd history.

Do not call the graph builder directly from `Tensor`. Forward through the Dispatcher so eager execution and graph recording follow the same path.

## Adding an optimizer

Derive from `Optimizer`, implement `step()`, and operate on `params_`. The existing SGD implementation validates gradient presence, shape, dtype, and device before delegating mutation to `Dispatcher::sgd`.

`Optimizer::zero_grad()` clears an existing gradient in place when safe. It does not create a gradient for a parameter that has not participated in backward.

## Adding a backend

The current implementation has only CPU storage and CPU execution. `DeviceType::CUDA` is a declaration, not a working backend.

A functional new device backend requires more than adding a kernel directory. At minimum it needs:

- device-specific storage allocation and deallocation;
- explicit transfer semantics;
- Dispatcher validation and routing for every supported operation;
- kernels for the advertised operation set;
- autograd behavior across supported device operations;
- CMake feature detection and conditional compilation;
- cross-device error handling and backend-specific tests.

Until those pieces exist, documentation and examples should continue to describe HELIX as CPU-only.

## Testing expectations

Add focused tests beside the affected subsystem:

- `tests/` for unit, integration, convergence, and gradient checks;
- `stress/` for lifetime, aliasing, extreme-shape, and concurrency behavior;
- `benchmark/` only when measuring performance rather than correctness.

Run the entire test suite after changes that affect Tensor layout, allocation, Dispatcher behavior, or autograd, because those subsystems share storage and traversal code.
