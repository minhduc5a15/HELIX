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

### 4. Why does the AVX2 Micro-kernel use a `4x16` Outer Product structure?

- **Reason:** Initially, an `8x1` Inner Product micro-kernel was used. However, it suffered from severe Read-Modify-Write (RMW) overhead in the innermost loop (`+=`), causing massive L1 Cache traffic and pipeline stalls. By switching to a `4x16` Outer Product pattern, we can hold the entire `4x16` block of Matrix C inside 8 YMM registers (256-bit). This completely eliminates intermediate memory writes to RAM until the entire K-loop finishes, pushing the CPU to 90+ GFLOPS.

### 5. Why was the Transpose of Matrix B (`B_T`) removed in the Dispatcher?

- **Reason:** Previously, `B` was transposed to `rhs_t` to ensure contiguous memory access. However, this caused unnecessary Memory Allocation overhead and latency. With the new `4x16` Outer Product AVX2 micro-kernel, we can broadcast elements of `A` and load contiguous rows of `B` directly using `_mm256_loadu_ps` without any transposition. This saves memory and speeds up the pipeline.

### 6. Why does Autograd compute In-place Gradients (`add_`)?

- **Reason:** In a large Deep Learning model, the Gradient of a Layer can receive derivatives from many branches in the computation graph (e.g., Residual Connections). If a new Tensor is created every time a Gradient is accumulated, the system would collapse due to Memory Allocation Overhead. Using `add_` (adding directly to old memory) completely eliminates the Memory Overhead of the Backward pass.

### 7. Why use a Thread-Local Memory Pool instead of a global `std::mutex`?

- **Reason:** When OpenMP threads were introduced for Matrix Multiplication, a central `global_free_blocks_` allocator protected by a `std::mutex` caused severe Lock Contention, degrading multi-threaded performance. By giving each thread its own lock-free `LocalPool` via `thread_local`, allocations became O(1) without any synchronization overhead. The `reset()` mechanism safely tracks thread IDs to prevent Memory Leaks.

### 8. Why use a JIT Profiler (AutoTuner) for Dispatching instead of a hardcoded threshold?

- **Reason:** OpenMP multi-threading only outperforms Single-Core AVX2 when the matrix is large enough to amortize thread spin-up overhead. However, this threshold varies wildly across hardware architectures (e.g., Intel vs AMD vs ARM) depending on Core Count and Cache Topology. Hardcoding a threshold like `128x128` caused catastrophic slowdowns on small matrices. The `AutoTuner` solves this by benchmarking the CPU at runtime (Hybrid Lazy Evaluation) and dynamically computing the exact FLOP threshold where OpenMP becomes profitable.

### 9. Why replace NDIterator with Chunked Iterator for Element-wise operations?

- **Reason:** The original `NDIterator` computed flat offsets element-by-element using N-dimensional modulo arithmetic. This scalar loop structure destroyed compiler auto-vectorization (AVX/SSE) and became a major bottleneck. The `Chunked Iterator` dynamically scans tensor metadata to find the deepest contiguous dimension (`chunk_dim`). It then extracts 1D segments (`chunk_size`) that can be perfectly unrolled and vectorized using `#pragma omp simd`, unlocking SIMD performance and achieving zero-overhead memory bloat on non-contiguous in-place operations.

### 10. Why does Autograd use `std::weak_ptr` in `AccumulateGrad`?

- **Reason:** In the dynamic computational graph, an `AccumulateGrad` node represents a leaf tensor that needs gradients. If this leaf tensor is destroyed before `.backward()` is called, the graph would traverse into freed memory, causing a fatal Use-After-Free (UAF) corruption. By holding a `std::weak_ptr` to the `AutogradMeta` instead of a raw pointer, `AccumulateGrad` can safely check if the memory is still alive (`lock()`). If it fails, it gracefully drops the gradient, guaranteeing memory safety without creating cyclic references that cause leaks.
