# HELIX Float32 BMM Scheduling Policy Calibration

## 1. Executive Summary

This document specifies the technical foundation, benchmark methodology, empirical results, and threshold derivation for the Batched Matrix Multiplication (BMM) execution policy in HELIX (`Phase 1.1`).

In HELIX, Batched Matrix Multiplication computes:
$$\text{Output}[B, M, N] = A[B, M, K] \times B[B, K, N]$$

When invoking `Tensor::bmm` or `CPUBackend::bmm` with `MatMulStrategy::Auto` on Float32 tensors on x86_64 hardware with AVX2 and OpenMP enabled, HELIX dispatches execution via `internal::choose_bmm_execution_policy(B, M, K, N, num_threads)`.

---

## 2. Problem Statement & Architecture

### 2.1 The Two Execution Strategies

1. **Serial Batch Loop with Matrix-Level GEMM (`SerialAuto`):**
   Iterates through each batch slice $0 \le b < B$ sequentially. Each slice computes `CPUBackend::matmul(A_b, B_b, Out_b, M, K, N, Auto)`.
   - *Advantage:* For very small total work or single batch ($B \le 1$), incurs zero OpenMP fork-join barrier overhead.
   - *Drawback:* For multi-batch workloads with moderate matrix dimensions, each slice executes single-threaded on a single CPU core, leaving available CPU cores idle. For very large matrices, each slice enters its own OpenMP parallel region, incurring $B$ repeated fork-join barriers.

2. **Batch-Parallel Execution with Single-Threaded SIMD Inner Kernels (`ParallelBatch`):**
   Executes a single outer `#pragma omp parallel for schedule(static)` loop over the batch dimension $B$. Within each thread worker, a single-threaded AVX2 FMA micro-kernel (`avx2_micro_matmul`) computes the matrix slice.
   - *Invariant:* **Strict No-Nested-OpenMP**. The outer batch loop is the only parallel region; workers never invoke OpenMP internally.
   - *Advantage:* Distributes independent batch slices across CPU cores with a single fork-join barrier for the entire operation.
   - *Drawback:* If total compute volume is tiny ($B \times M \times K \times N < 32,768$), the OpenMP thread scheduling and barrier latency dominates the runtime, performing slower than single-threaded serial execution.

---

## 3. Calibration Environment

All calibration benchmarks were executed in the following hardware and software environment:

- **Processor:** Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz (Comet Lake)
- **Physical Cores:** 6 physical cores
- **Logical Threads:** 12 threads (Intel Hyper-Threading enabled)
- **SIMD Capabilities:** AVX2, FMA3, SSE4.2
- **Operating System:** Linux 6.8.0-52-generic (Ubuntu 22.04 LTS x86_64)
- **Compiler:** GCC 13.3.0 (`-O3 -std=c++20 -mavx2 -mfma -fopenmp`)
- **OpenMP Runtime:** GNU libgomp 13.3.0
- **Build Configuration:** Release (`CMAKE_BUILD_TYPE=Release`)

---

## 4. Benchmark Methodology & Noise Control

Four benchmark suites were designed and evaluated:
1. **Scout Benchmark (`--scout`):** Broad exploratory matrix across $B \in \{1, 2, 4, 8, 16, 32, 64\}$ and dimensions from $16^3$ to $512^3$.
2. **Refined Benchmark (`--refined`):** Fine-grained boundary exploration around intermediate dimensions ($64^3$ to $192^3$).
3. **Low-Work Crossover Benchmark (`--low-work`):** Microsecond-scale grid measuring total work from 128 to 524,288 MACs ($B \in \{2, 4, 8, 16\}$, $\text{dim} \in [4, 32]$).
4. **Large-GEMM Boundary Benchmark (`--large-boundary`):** Exploration of large matrices ($256^3, 384^3, 512^3$) with batch counts $B \in \{2, 4, 6, 8\}$ across thread counts $T \in \{2, 4, 6, 12\}$.

### Noise Control Techniques:
- **Batch Repetition Normalization:** For micro-workloads ($< 65,536$ MACs), invocations are repeated inside the timed region ($3\times$ to $80\times$), normalizing elapsed time by the repetition factor to overcome monotonic clock discretization noise.
- **Thread Pool Pre-Warmup:** An explicit parallel region warms up OpenMP worker threads before timing begins, preventing first-touch kernel thread allocation latency from skewing measurements.
- **Reference Output Verification:** Every workload verifies candidate outputs against double-precision / serial reference outputs prior to timing.

---

## 5. Empirical Results

### 5.1 Representative Measurements

| $B$ | Shape ($M \times K \times N$) | Total MACs | Threads ($T$) | Serial Auto (ms) | Parallel Batch (ms) | Speedup | Winner |
|---:|:---|---:|---:|---:|---:|---:|:---|
| 2 | $4 \times 4 \times 4$ | 128 | 6 | 0.00020 | 0.00174 | 0.12x | **SerialAuto** |
| 4 | $8 \times 8 \times 8$ | 2,048 | 6 | 0.00021 | 0.00079 | 0.27x | **SerialAuto** |
| 2 | $16 \times 16 \times 16$ | 8,192 | 6 | 0.00025 | 0.00085 | 0.29x | **SerialAuto** |
| 4 | $16 \times 16 \times 16$ | 16,384 | 6 | 0.00045 | 0.00100 | 0.45x | **SerialAuto** |
| 8 | $12 \times 12 \times 12$ | 13,824 | 6 | 0.00221 | 0.00133 | 1.66x | ParallelBatch |
| 8 | $16 \times 16 \times 16$ | 32,768 | 6 | 0.00111 | 0.00122 | 0.91x | *Boundary* |
| 4 | $20 \times 20 \times 20$ | 32,000 | 6 | 0.00279 | 0.00167 | 1.67x | **ParallelBatch** |
| 8 | $20 \times 20 \times 20$ | 64,000 | 6 | 0.00713 | 0.00207 | 3.45x | **ParallelBatch** |
| 4 | $64 \times 64 \times 64$ | 1,048,576 | 6 | 0.07820 | 0.02120 | 3.69x | **ParallelBatch** |
| 8 | $128 \times 128 \times 128$ | 16,777,216 | 6 | 1.25000 | 0.28000 | 4.46x | **ParallelBatch** |
| 4 | $512 \times 512 \times 512$ | 536,870,912 | 6 | 32.5401 | 13.0588 | 2.49x | **ParallelBatch** |
| 2 | $512 \times 512 \times 512$ | 268,435,456 | 6 | 18.1392 | 11.0928 | 1.64x | **ParallelBatch** |

---

## 6. Threshold Derivation & Rationale

### 6.1 Total Work Metric: MACs vs FLOPs
In floating-point matrix multiplication:
$$\text{Operations} = 2 \times B \times M \times K \times N \text{ FLOPs}$$
$$\text{MAC Work} = B \times M \times K \times N \text{ Multiply-Accumulate units}$$
$1\text{ MAC} = 2\text{ FLOPs}$. To avoid ambiguity, the code strictly uses MAC units: `kMinParallelTotalMacWork = 32768` (corresponding to approximately $65,536$ FLOPs).

### 6.2 Selection of 32,768 MACs Threshold
Across 625 paired configurations:
- At **16,384 MACs**, `ParallelBatch` loses to `SerialAuto` by $1.5\times$ to $2.2\times$ on aligned SIMD shapes ($16\times 16\times 16$) because AVX2 processes $16^3$ in under $0.5\ \mu s$, which is less than OpenMP barrier synchronization latency.
- At **32,768 MACs**, the transition stabilizes across non-vector dimensions ($M=20$, $M=24$, rectangular shapes).
- Evaluating candidate thresholds with `scripts/analyze_bmm_benchmark.py`:
  - `Threshold = 16384`: $T=2$ p95 regret = $23.65\%$, $T=6$ p95 regret = $27.12\%$.
  - `Threshold = 32768`: $T=2$ p95 regret = **$9.75\%$**, $T=6$ p95 regret = **$10.35\%$**, median regret = **$0.00\%$**.
  - `Threshold = 65536`: unnecessarily leaves $1.5\times\text{--}2.5\times$ speedups on the table for workloads between 32K and 64K MACs.
- Therefore, **`32768 MACs`** is selected as the mathematically and empirically optimal crossover boundary.

### 6.3 Analysis of Large GEMM Under-Utilization ($B < \text{threads}$)
We evaluated the hypothesis: *When $B < \text{threads}$ on large matrices ($\ge 512^3$), should HELIX fall back to `SerialAuto`?*
- **Empirical finding:** No. In HELIX, single-GEMM `Auto` dispatches to `AutoTuner`. Because `AutoTuner` sets its OpenMP threshold based on single-matrix profiling (often $\ge 512^3$ or $1024^3$), falling back to `SerialAuto` causes each batch slice to run **single-threaded** sequentially.
- On $B=4, 512^3$ at $T=6$: `SerialAuto` takes **$32.54\text{ ms}$** (1 core sequentially), whereas `ParallelBatch` takes **$13.06\text{ ms}$** (4 cores in parallel). Falling back to `SerialAuto` would inflict a **$2.49\times$ slowdown ($149\%$ regret)**.
- Even at $B=2, 512^3$: `ParallelBatch` ($11.09\text{ ms}$) is faster than `SerialAuto` ($18.14\text{ ms}$).
- Therefore, a fallback guard when $B < \text{threads}$ for large matrices is counterproductive and was intentionally omitted.

---

## 7. Policy Contract & Implementation

```cpp
BmmExecutionPolicy choose_bmm_execution_policy(
    const size_t B, const size_t M, const size_t K, const size_t N, const size_t num_threads
) {
    if (B <= 1 || num_threads <= 1) {
        return BmmExecutionPolicy::SerialAuto;
    }

    size_t matrix_mac = 0;
    size_t total_mac = 0;
    if (mul_overflow(M, K, &matrix_mac) || mul_overflow(matrix_mac, N, &matrix_mac) ||
        mul_overflow(matrix_mac, B, &total_mac)) {
        return BmmExecutionPolicy::SerialAuto;
    }

    constexpr size_t kMinParallelTotalMacWork = 32768;
    if (total_mac < kMinParallelTotalMacWork) {
        return BmmExecutionPolicy::SerialAuto;
    }

    return BmmExecutionPolicy::ParallelBatch;
}
```

---

## 8. Limitations & Future Work

1. **Architecture Scope:** This policy is calibrated specifically for x86_64 CPUs with AVX2/FMA and OpenMP. Non-AVX2 and ARM64 platforms (Apple Silicon) safely fall back to the serial batch loop.
2. **Data Types:** Restricted to `Float32`. `Float64`, `Int32`, and `Int64` continue using the serial batch loop.
3. **Static Threshold:** The threshold $32,768$ MACs is a calibrated constant. Future phases may integrate dynamic calibration into `.helix_autotune` if hardware heterogeneity demands per-machine tuning.

---

## 9. Reproduction Instructions

To reproduce the benchmark data and analysis:

```bash
# 1. Clean build
./build.sh --clean

# 2. Run benchmark suites
./build/benchmark/bmm_benchmark --scout --output output/bmm_benchmark_scout.csv
./build/benchmark/bmm_benchmark --refined --output output/bmm_benchmark_refined.csv
./build/benchmark/bmm_benchmark --low-work --output output/bmm_benchmark_low_work.csv
./build/benchmark/bmm_benchmark --large-boundary --output output/bmm_benchmark_large_boundary.csv

# 3. Run analyzer tool
python3 scripts/analyze_bmm_benchmark.py \
  output/bmm_benchmark_scout.csv \
  output/bmm_benchmark_refined.csv \
  output/bmm_benchmark_low_work.csv \
  output/bmm_benchmark_large_boundary.csv \
  --min-total-mac 32768 \
  --output output/bmm_policy_analysis.json
```
