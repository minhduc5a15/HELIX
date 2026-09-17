# Design Decisions

This document explains decisions embodied in the current implementation. Performance statements are limited to mechanisms visible in the code; machine-specific measurements belong in the benchmark output.

## Dispatcher between Tensor and kernels

`Tensor` exposes the user-facing API, while `Dispatcher` handles broadcasting, dtype promotion, layout preparation, kernel selection, and graph-recording callbacks.

This keeps operation policy out of the value-like Tensor handle and gives eager execution and autograd one common path. It also provides a place for future device routing, although only CPU execution is implemented today.

## Dynamic computation graph

HELIX builds the autograd graph as operations execute. This matches ordinary C++ control flow: loops and branches determine the graph produced by that run.

Each differentiable forward operation records a `Node` only when at least one input requires gradients. Backward uses a topological schedule so a shared ancestor receives all branch contributions before its backward function executes.

## Saved values share storage and carry a version

Backward formulas need some forward values. Copying every saved tensor would be expensive, while retaining the original tensor with its complete autograd metadata could create ownership cycles.

`SavedTensor` therefore stores a detached tensor that shares storage and records the storage version. An in-place mutation increments that version, and unpacking the saved value during backward then raises an error instead of silently using changed data.

## Weak ownership for leaf gradient accumulation

An `AccumulateGrad` node can outlive the leaf tensor whose gradient it would update. It stores a `weak_ptr<AutogradMeta>` so it can detect that the leaf metadata has expired without keeping a reference cycle alive or dereferencing freed memory.

## Conditional in-place gradient accumulation

Gradients from multiple graph branches must be summed. The engine and `AccumulateGrad` use `add_` only when the destination gradient is unshared, shape-compatible, non-overlapping, and of the required dtype. Otherwise they allocate a safe out-of-place sum or clone.

This optimization reduces allocations when ownership and layout make mutation safe. It does not guarantee that every backward accumulation is allocation-free.

## Shared storage for views

`view`, `slice`, `transpose`, and `broadcast_to` represent layout changes with shape, stride, and storage-offset metadata. Their backward nodes implement the corresponding inverse or reduction behavior.

`reshape` is zero-copy only for contiguous input. For non-contiguous input it clones before creating the requested view. Broadcast views use zero strides and are rejected as destinations for supported in-place mutations because multiple logical indices can refer to one storage element.

## TensorIterator for general binary traversal

Contiguous binary operations use a direct dense kernel. Other binary layouts use `TensorIterator`, which computes broadcast strides, coalesces compatible dimensions, and traverses fixed-size chunks in parallel when the workload is large enough.

Legacy `NDIterator` helpers remain in use for several copy, zeroing, in-place, and optimizer paths. The project has not replaced every strided traversal with `TensorIterator`.

## Cache blocking before SIMD and threading

The matrix-multiplication backends preserve separate naive, blocked, AVX2, and OpenMP strategies. The naive kernel is a correctness and measurement baseline. Blocking improves locality and also provides a portable fallback for types without a specialized SIMD kernel.

The float AVX2 implementation uses a 4-by-16 outer-product micro-kernel. Eight YMM accumulators keep a 4-by-16 output tile in registers across the K loop, and smaller kernels or scalar loops handle tails. Matrix B remains in row-major order; the kernel loads contiguous segments from its rows rather than materializing a transposed copy.

## Runtime threshold calibration for OpenMP matmul

OpenMP startup and synchronization can cost more than they save for small matrices. A single hardcoded threshold would behave differently across CPUs and OpenMP runtimes.

`AutoTuner` compares the float AVX2 and OpenMP kernels at square sizes 512 and, when useful, 256. It chooses one of three volume thresholds: `256^3`, `512^3`, or `1024^3`. The value is cached in `.helix_autotune` and loaded on later runs from the same working directory.

This is a small runtime calibration, not an exhaustive search for an exact crossover point. It calibrates only automatic float matrix-multiplication dispatch.

## Thread-local allocation caches with synchronized global bins

The allocator rounds allocations to 32-byte boundaries and keeps freed blocks by size. The common local-cache path does not need a global lock. Global bins use mutexes for refill batches and blocks returned across thread boundaries.

An epoch lets `reset()` request lazy cleanup of active thread caches without directly freeing blocks owned by running threads. The singleton itself intentionally remains alive until process termination so thread-local destructors cannot access a destroyed pool during shutdown.

## CPU-only execution boundary

`DeviceType` includes `CUDA`, but the runtime has no CUDA allocator, transfers, or kernels. Keeping the enum allows the API to express a future direction; it does not constitute backend support. Current execution paths are written for CPU kernels, and new operations must validate device combinations explicitly.
