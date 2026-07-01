# Coding Convention

HELIX uses the C++20 standard. To keep the project consistent and maintainable, all code contributions (Pull Requests) must adhere to the following guidelines.

## 1. Naming Convention

- **Class / Struct**: PascalCase (e.g., `Tensor`, `AddNode`, `Sequential`).
- **Function / Method**: snake_case (e.g., `backward()`, `register_parameter()`, `forward()`).
- **Variable**: snake_case (e.g., `input_tensor`, `learning_rate`).
- **Constant / Macro / Enum values**: UPPER_SNAKE_CASE (e.g., `HELIX_USE_FMA`, `BLOCK_SIZE`).
- **Private Data Members**: Must have an underscore suffix `_` (e.g., `shape_`, `stride_`, `grad_fn_`).
- **Namespace**: All source code belongs to the `namespace helix`. Sub-namespaces are `core`, `autograd`, `backend`, `nn`, `optim`.

## 2. File Layout

A standard `.cpp` or `.hpp` file should be organized with Includes in the following order:

```cpp
// 1. The corresponding Header for this .cpp file (If it's a .cpp file)
#include "my_header.hpp"

// 2. C++ Standard Library (Sorted alphabetically)
#include <iostream>
#include <memory>
#include <vector>

// 3. Third-party Libraries
// (None currently in HELIX)

// 4. Other HELIX Headers
#include "core/tensor.hpp"
#include "autograd/node.hpp"
```

## 3. Const Correctness

- Any variable that does not change its value must be marked as `const`.
- Any member function (Method) that does not modify the object's state (e.g., getters like `shape()`, `stride()`, and especially the `forward()` function of a Neural Network class if it doesn't store internal state) must end with `const`.
- Pass large objects (e.g., `std::vector`, `std::string`) by Const Reference `const T&`. For `std::shared_ptr<Tensor>`, if ownership is not transferred, pass-by-value is acceptable as the pointer copy cost is extremely cheap.

## 4. Pointers and Memory Management
- **ABSOLUTELY NO** raw pointers (`new` or `delete`) for resources living outside the function scope.
- All Tensor and Autograd Node objects are managed using `std::shared_ptr`.
- When needing to prevent Circular Dependency in the Autograd Graph, use `std::weak_ptr` or non-owning raw pointers if the lifecycle is guaranteed.

## 5. Comments and Code Documentation
- **Public APIs**: Must have standard Doxygen comments describing `@brief`, `@param`, `@return`.
- **In-code Comments**: Only comment to explain **WHY** you wrote this code, rather than describing **WHAT** it does. (e.g., Don't write `// Iterate through matrix`, write `// Iterate column-wise to utilize Hardware Prefetcher and avoid Cache Misses`).
