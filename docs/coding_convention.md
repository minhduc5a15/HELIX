# Coding Convention

HELIX is compiled as C++20 and formatted with the checked-in `.clang-format`. This document records conventions visible in the current codebase and requirements that keep new code compatible with it.

## Naming

- Classes and structs use `PascalCase`: `Tensor`, `TensorIterator`, `BackwardEngine`.
- Functions, methods, and variables use `snake_case`: `compute_broadcast_shape`, `learning_rate`.
- Private data members use a trailing underscore: `shape_`, `grad_fn_`.
- Compile-time constants use either a descriptive `kPascalCase` local name or the existing uppercase style used by subsystem-wide constants. Match the surrounding file.
- Public framework APIs live in `namespace helix`. Benchmark support code lives in `namespace helix::benchmark`. Directory names such as `core/`, `nn/`, and `optim/` do not imply matching C++ namespaces.

## Formatting and includes

Run the repository formatter when changing C++ sources:

```bash
clang-format -i path/to/changed_file.cpp path/to/changed_file.hpp
```

For a `.cpp` file, include its corresponding project header first when one exists, followed by standard-library headers, platform or third-party headers, and other HELIX headers. Let `clang-format` preserve the final ordering and layout.

Use `#pragma once` in headers, as existing headers do.

## Interfaces and ownership

- Pass read-only tensors and other nontrivial objects by `const&` unless a copy is intentional.
- Use `std::move` when transferring ownership into a member or container.
- Use `std::shared_ptr` where ownership is shared, such as `TensorImpl`, `Storage`, module objects held by `Sequential`, and autograd nodes.
- Use `std::weak_ptr` to break ownership cycles or observe an object that may expire, as `AccumulateGrad` does for leaf metadata.
- Raw pointers are appropriate for non-owning kernel buffers and explicitly non-owning registries. Make ownership and lifetime visible in the surrounding interface.
- Resource-owning code should use RAII. Any deliberate process-lifetime allocation must explain the teardown constraint that requires it.

## Const correctness and types

- Mark local values `const` when they are not reassigned and doing so keeps the code readable.
- Mark member functions `const` when they do not modify logical object state.
- Use `size_t` for shapes and element counts. Use `ptrdiff_t` for strides, offsets, and OpenMP loop variables that may need signed arithmetic.
- Do not assume tensor data is `float`. Dispatch on `DType` and use `data_ptr<T>()` unless an API is intentionally restricted to `Float32`.
- Check shape, dtype, layout, device, and overlap assumptions before entering a low-level kernel.

## Errors

Use exceptions consistently with the existing API:

- `std::invalid_argument` for incompatible values such as shapes or dtypes;
- `std::out_of_range` for invalid axes or indices;
- `std::runtime_error` for unsupported execution paths or invalid runtime state;
- `std::overflow_error` when shape arithmetic cannot be represented.

Error messages should name the failed operation and the violated condition.

## Comments and documentation

Comments should explain invariants, ownership, numerical reasoning, or a non-obvious performance decision. Avoid claims about speed or memory behavior unless a benchmark or test in the repository supports them.

Document public APIs when adding them. Keep examples compilable against the current interface, and state limitations such as supported rank, dtype, target representation, or device.

## Verification

Build with warnings enabled and run the relevant tests. For changes to shared runtime code, run the complete suite:

```bash
./build.sh --debug
./run_tests.sh
```

Use numerical gradient checks for new differentiable operations. Use benchmarks to measure performance changes, but do not turn a machine-specific result into a general performance guarantee.
