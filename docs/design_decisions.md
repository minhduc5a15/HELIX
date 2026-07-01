# Design Decisions

This document records the core design decisions of HELIX and **why** they were chosen. This is the most important document for a Developer to understand the technical philosophy of the project.

---

### 1. Why use a Dispatcher instead of Hard-coding in Tensor?
Instead of placing the logic to add two Tensors directly into the `Tensor` class, we separated it into `Dispatcher` -> `Backend`.
- **Reason:** Scalability. As HELIX grows and supports GPUs (CUDA, Vulkan), the `Tensor` class would bloat uncontrollably if it had to encapsulate both CPU and GPU logic. The `Dispatcher` keeps the `Tensor Runtime` architecture completely independent of the underlying physical hardware.

### 2. Why choose a Dynamic Computational Graph (Define-by-Run)?
HELIX learns from PyTorch's Autograd architecture (Dynamic Graph) rather than TensorFlow 1.x (Static Graph).
- **Reason:** Flexibility and Debuggability. The graph is generated at C++ code execution time, allowing users the freedom to use host language `for` loops and `if/else` statements. Developer Experience (DX) is crucial for a Framework.

### 3. Why optimize Cache Blocking first, then AVX2?
During the development of the MatMul (Matrix Multiplication) Backend, the optimization roadmap was sequentially: Naive -> Blocked -> AVX2 -> OpenMP.
- **Reason:** The Cache Memory Access Pattern accounts for 80% of matrix computation performance. AVX2 is only effective when data is loaded continuously. Implementing Cache Blocking helps verify if there are Cache Misses, creating a clean data loading pipeline for the Hardware Prefetcher before launching AVX2 Vector instructions.

### 4. Why does the AVX2 Micro-kernel use an `8x1` structure (And will move to larger structures)?
- **Reason:** The `8x1` micro-kernel is the simplest test of Column-wise Vectorization, allowing maximum utilization of L1 Cache bandwidth without the headache of complex Memory Packing or deep Loop Unrolling. However, it hits an Arithmetic Intensity limit at `0.88 FMA/Load`. Therefore, the architecture is open to easily supplementing `4x4` or larger Micro-kernels in the future to boost performance.

### 5. Why is `b_t` (B Transpose) passed into MatMul instead of `B`?
- **Reason:** This is an extremely effective optimization trick. Transposing `B` to `B_T` right at the `Tensor` layer ensures that the innermost loop of all matrix multiplication algorithms (both Naive and Blocked) accesses arrays completely linearly (Contiguous Array). Consequently, the traditional Cache Miss problem of MatMul is halved right before the algorithm even starts.

### 6. Why does Autograd compute In-place Gradients (`add_`)?
- **Reason:** In a large Deep Learning model, the Gradient of a Layer can receive derivatives from many branches in the computation graph (e.g., Residual Connections). If a new Tensor is created every time a Gradient is accumulated, the system would collapse due to Memory Allocation Overhead. Using `add_` (adding directly to old memory) completely eliminates the Memory Overhead of the Backward pass.
