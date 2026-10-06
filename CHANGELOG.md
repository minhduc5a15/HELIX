# Changelog

This file records notable user-visible and developer-facing changes to HELIX. Its release dates and version boundaries follow the Git tags in this repository. The structure is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Added `no_grad` RAII scope guard with thread-local autograd suppression, exception safety, and support for nested scopes.
- Added `Tensor::argmax(size_t dim)` returning an `Int64` index tensor along a dimension (scalar for rank-1 tensors) supporting all dtypes and strided layouts.
- Added `helix::cat` along any dimension with layout validation, overflow checks, fast contiguous block copying, and dynamic reverse-mode autograd (`CatBackward`).
- Added `tanh` to the Tensor API, CPU kernels, autograd graph builder, backward functions, and gradient tests.
- Added range-based `Tensor::flatten(start_dim, end_dim)` and functional `helix::flatten(input, start_dim, end_dim)` with negative indexing, overflow validation, zero-size extents, non-contiguous layouts, and autograd.
- Added zero-copy metadata-only views for Tensor methods and functional APIs: `Tensor::unsqueeze(dim)`, `Tensor::squeeze()`, `Tensor::squeeze(dim)`, `helix::unsqueeze(input, dim)`, `helix::squeeze(input)`, and `helix::squeeze(input, dim)`, preserving storage sharing, offset, and non-contiguous stride layouts with checked stride overflow protection and `ViewBackward` autograd integration.
- Added `helix::Flatten` module defaulting to `start_dim=1, end_dim=-1` for batch preservation in neural networks and `Sequential`.
- Added batched matrix multiplication (`bmm`) supporting strict 3D tensors (`[B, M, K]` x `[B, K, N]` -> `[B, M, N]`) with `Tensor::bmm` and `helix::bmm`, multi-dtype promotion, non-contiguous layouts, zero-extent handling, and reverse-mode autograd (`BmmBackward`) with `SavedTensor` in-place version safety.
- Added autograd support for `clone`, `view`, `slice`, `transpose`, and `broadcast_to`.
- Added operational support for `Float32`, `Float64`, `Int32`, and `Int64` across tensor factories, element-wise operations, reductions, matrix multiplication, and dtype promotion. Autograd remains restricted to floating-point tensors.
- Added `TensorIterator` for broadcast-aware binary traversal with dimension coalescing, fixed-size traversal state, and parallel chunking.
- Added the logic-gates MLP example.
- Added ThreadSanitizer build support and stress coverage for allocation lifetime, graph teardown, in-place mutation, overlapping strides, extreme shapes, and OpenMP concurrency.
- Added cross-platform helpers for checked size multiplication and AVX2/FMA capability detection.

### Changed

- Routed supported view operations through `Dispatcher` so eager execution and graph recording use the same path.
- Added storage version tracking and `SavedTensor` checks so backward rejects saved values changed by in-place operations.
- Changed leaf accumulation to hold `AutogradMeta` through `weak_ptr`, avoiding dangling references when a leaf is destroyed before backward.
- Made gradient accumulation reuse storage only when ownership, dtype, shape, contiguity, and overlap checks permit safe mutation.
- Added `retain_graph` behavior and graph-edge cleanup after backward; a second backward through a released graph now reports an error.
- Reworked the memory pool with epoch-based lazy cache reclamation and process-lifetime ownership to avoid races between reset, thread-local teardown, and detached threads.
- Changed stride and iterator offset arithmetic to signed `ptrdiff_t` where negative strides must be represented.
- Optimized non-contiguous clone, copy, zeroing, in-place addition, and SGD traversal, including specialized 2D paths.
- Optimized AVX2 scalar tails, contiguous copies, and OpenMP matrix-multiplication initialization and loop bounds.
- Expanded the MNIST example to a `784 -> 256 -> 128 -> 10` network, batch size 64, and ten training epochs, utilizing `Tensor::argmax` for accuracy and `no_grad` for test evaluation.
- Standardized local build output under `build/`.
- Updated README and all Markdown documents under `docs/` to match the current API, implemented backends, iterator architecture, and benchmark methodology.

### Fixed

- Rejected shape products and contiguous-stride calculations that overflow `size_t`.
- Fixed use-after-free and shutdown-order bugs in autograd metadata and the memory pool.
- Fixed races and unsafe mutation involving broadcast or internally overlapping tensor views.
- Fixed graph leaks, repeated-backward corruption, and unbounded gradient accumulation across training iterations.
- Fixed missing graph construction for scalar arithmetic operations.
- Fixed dtype erasure and gradient truncation in tensor factories, casts, ReLU backward, slice backward, leaf accumulation, and gradient clearing.
- Fixed `copy_` behavior for dtype conversion, aliasing, self-copy, zero-sized tensors, and differently shaped tensors with the same element count.
- Fixed OpenMP signed-loop compatibility issues on MSVC and related macOS/Windows build problems.
- Fixed `TensorIterator` rank handling, broadcast coalescing, scalar broadcast, high-rank contiguous fast paths, and zero-sized tensor traversal.
- Fixed allocator memory accounting underflow on thread cache and global bin allocation hits.
- Fixed stale epoch block resurrection in `ThreadCacheWrapper` teardown after `MemoryPool::reset()`.
- Fixed broken row-load SIMD dot product in `avx2_dot_matmul` by delegating to `avx2_micro_matmul`.
- Fixed division-by-zero crashes in `NDIterator` and `BinaryNDIterator` when indexing zero-sized tensors.
- Fixed signed stride truncation and unsigned overflow on negative strides in 2D in-place `add_` and `sgd` kernels.
- Fixed data races in `AutoTuner::calibrate` and `AutoTuner::get_omp_threshold` using double-checked atomic synchronization.
- Fixed thread safety race in `TensorFactory::randn` by using a `thread_local` Mersenne Twister engine.
- Added cycle detection in `BackwardEngine::run` to reject cyclic computation graphs with a runtime exception.
- Added coordinate bounds validation in `Tensor::item` and `Tensor::set_item`.
- Added dimension validation in `Dispatcher::cross_entropy` to reject empty batch or class dimensions.
- Fixed singleton memory leak on module unload in `AutogradEngineProvider` and `AutogradGraphBuilderProvider`.
- Fixed forward-overlapping in-place self-copy data corruption in `Tensor::copy_` by cloning aliased sources.
- Fixed type safety crash in rank-2 non-contiguous `Tensor::zero_()` by dispatching across all primitive dtypes.
- Prevented uncatchable `SIGABRT` / `std::terminate` process aborts in OpenMP parallel regions by rejecting tensors with rank > 8 prior to parallel thread spawning.
- Fixed hardware trap `SIGFPE` (integer division-by-zero) in `BinaryNDIterator::compute_offset_from_flat` on zero-sized shapes.
- Fixed algorithmic work-sharing breakdown and duplicate writes in `Tensor::zero_()` OpenMP fallback by dividing elements into disjoint `[start, end)` chunks.
- Fixed unchecked integer multiplication wraparound in `TensorImpl` storage allocation by adding checked size arithmetic against `SIZE_MAX`.
- Guarded `CrossEntropyLoss` against differentiable targets during autograd graph construction to prevent invalid shape-mismatched optimizer states.
- Fixed unsigned integer conversion of negative strides in `Stride::compute_offset` and `Dispatcher::slice` by explicitly casting to signed `ptrdiff_t`.
- Removed dead `all_caches_` container and `caches_mutex_` from `MemoryPool`, eliminating lock contention during worker thread initialization and teardown.
- Fixed uninitialized output memory in `openmp_matmul` fallback when $K = 0$ on non-AVX2 hardware targets.

### Removed

- Removed the obsolete Doxygen configuration and bundled Doxygen theme files.

## [1.2.0] - 2026-08-07

### Added

- Added the end-to-end MNIST example, IDX data loader, one-hot label conversion, download script, batched training loop, and evaluation pass.
- Added stateful `NDIterator` and `BinaryNDIterator` helpers for strided traversal.
- Added chunk-based traversal that identifies a contiguous dimension for clone, `copy_`, `zero_`, in-place addition, and SGD paths.

### Changed

- Replaced repeated full offset recomputation in several non-contiguous operations with incremental iterator traversal.
- Reduced OpenMP overhead for small matrix multiplications.
- Reduced temporary allocation in the MNIST data loader.

### Fixed

- Fixed thread-local memory-pool teardown crashes and use-after-free conditions.
- Hardened chunk traversal for aliasing, broadcasted views, differently shaped copies, scalar tails, and zero-sized tensors.

## [1.1.1] - 2026-08-03

### Added

- Added `cross_entropy_loss` for 2D logits and same-shaped one-hot targets.
- Added a fused CPU cross-entropy kernel using log-sum-exp stabilization and its autograd backward node.
- Added cross-entropy value, gradient, invalid-shape, extreme-logit, non-contiguous-input, and training tests.
- Extended neural-network and training benchmarks with cross-entropy workloads.

### Changed

- Removed the 2048-by-2048 case from the default matrix benchmark set.
- Standardized remaining source comments in English.

## [1.1.0] - 2026-07-30

### Added

- Added a 32-byte-aligned caching allocator with thread-local caches, synchronized global size bins, and batched transfers between them.
- Added `AutoTuner`, including lazy or explicit calibration, an on-disk `.helix_autotune` cache, and tests for cache creation and loading.
- Added allocator stress tests for cross-thread deallocation and thread shutdown.

### Changed

- Replaced the earlier float matrix-multiplication inner-product path with a 4-by-16 AVX2/FMA outer-product micro-kernel and tail handling.
- Changed the float matrix-multiplication dispatcher to choose between blocked, AVX2, and OpenMP paths, using the calibrated compute-volume threshold when OpenMP is available.
- Changed the AVX2/OpenMP matrix kernels to consume row-major B directly instead of creating a transposed copy.
- Reworked the OpenMP matrix loop structure and benchmark documentation for the new kernels.

### Fixed

- Fixed a matrix-multiplication gradient-check tolerance issue on Windows.
- Fixed compilation on macOS configurations without OpenMP.

## [1.0.0] - 2026-07-10

### Added

- Added the C++20 Tensor runtime with shape, stride, shared storage, broadcasting, slicing, transposition, contiguous views, reshape-with-copy fallback, cloning, reductions, and element-wise and scalar operations.
- Added dynamic reverse-mode autograd for the initial Tensor operation set, including topological backward traversal and leaf-gradient accumulation.
- Added `Module`, `Linear`, `ReLU`, `Sequential`, the `mse_loss` free function, `Optimizer`, and SGD.
- Added naive, cache-blocked, AVX2, and OpenMP float matrix-multiplication strategies with runtime CPU dispatch.
- Added linear-regression and XOR training examples.
- Added the custom timing and CSV benchmark infrastructure for matrix multiplication, tensor operations, neural-network execution, and training loops.
- Added GoogleTest unit, numerical-gradient, integration, convergence, and initial stress tests.
- Added CMake build configuration, helper scripts, and GitHub Actions build and evaluation workflows.
- Added the initial architecture, design-decision, developer, benchmark, and coding-convention documents.

### Changed

- Refactored the early implementation into separate tensor, autograd, backend, neural-network, optimizer, benchmark, example, test, and stress-test directories.
- Separated matrix-multiplication strategies behind `CPUBackend` and the Dispatcher.
