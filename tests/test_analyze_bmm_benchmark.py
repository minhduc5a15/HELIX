#!/usr/bin/env python3
"""
Unit tests for scripts/analyze_bmm_benchmark.py
Validates boundary conditions, 64-bit overflow simulation, C++ parity test vectors,
duplicate aggregation contracts, and report generation.
"""

import csv
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path so we can import from scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_bmm_benchmark import (
    UINT64_MAX,
    simulate_policy,
    load_and_validate_csvs,
    analyze_configurations,
    print_text_report,
    print_markdown_report,
)


class TestAnalyzeBmmBenchmark(unittest.TestCase):

    def test_simulate_policy_boundaries(self):
        """
        Verify boundary conditions for B, threads, and total MAC work.
        Coverage:
          B = 0, 1, 2, 8
          threads = 0, 1, 2, 6, 12
          total_mac < 32768, = 32768, > 32768
        """
        # Outer parallelism unavailable: B <= 1 or threads <= 1
        self.assertEqual(simulate_policy(b=0, m=128, k=128, n=128, threads=8), "serial_auto")
        self.assertEqual(simulate_policy(b=1, m=128, k=128, n=128, threads=8), "serial_auto")
        self.assertEqual(simulate_policy(b=8, m=128, k=128, n=128, threads=0), "serial_auto")
        self.assertEqual(simulate_policy(b=8, m=128, k=128, n=128, threads=1), "serial_auto")

        # total_mac < 32768 (e.g. B=4, 16x16x16 -> 4 * 4096 = 16384)
        self.assertEqual(simulate_policy(b=4, m=16, k=16, n=16, threads=4), "serial_auto")
        self.assertEqual(simulate_policy(b=2, m=16, k=16, n=16, threads=4), "serial_auto")
        self.assertEqual(simulate_policy(b=4, m=20, k=20, n=20, threads=4), "serial_auto")  # 4 * 8000 = 32000 < 32768

        # total_mac == 32768 (e.g. B=8, 16x16x16 -> 8 * 4096 = 32768)
        self.assertEqual(simulate_policy(b=8, m=16, k=16, n=16, threads=4), "parallel_batch_serial_inner")

        # total_mac > 32768
        self.assertEqual(simulate_policy(b=8, m=20, k=20, n=20, threads=4), "parallel_batch_serial_inner")  # 64000
        self.assertEqual(simulate_policy(b=8, m=32, k=32, n=32, threads=4), "parallel_batch_serial_inner")
        self.assertEqual(simulate_policy(b=16, m=64, k=64, n=64, threads=6), "parallel_batch_serial_inner")

    def test_simulate_policy_large_matrix_guard(self):
        """
        Verify large-matrix guard behavior (matrix_mac >= 512^3 when B < threads).
        Default (matching C++ production): disabled -> parallel_batch_serial_inner.
        Enabled: enabled -> serial_auto.
        """
        # B=2, threads=4, matrix_mac = 512^3 = 134,217,728
        # Without guard (production default)
        self.assertEqual(
            simulate_policy(b=2, m=512, k=512, n=512, threads=4, enable_large_guard=False),
            "parallel_batch_serial_inner",
        )
        # With guard enabled
        self.assertEqual(
            simulate_policy(b=2, m=512, k=512, n=512, threads=4, enable_large_guard=True),
            "serial_auto",
        )
        # With guard enabled but B >= threads -> parallel_batch_serial_inner
        self.assertEqual(
            simulate_policy(b=4, m=512, k=512, n=512, threads=4, enable_large_guard=True),
            "parallel_batch_serial_inner",
        )

    def test_simulate_policy_overflow_handling(self):
        """
        Verify 64-bit unsigned integer mul_overflow simulation.
        Any overflow across M*K, (M*K)*N, or matrix_mac*B must fall back to serial_auto.
        """
        # M * K overflows UINT64_MAX
        self.assertEqual(simulate_policy(b=2, m=UINT64_MAX, k=2, n=2, threads=4), "serial_auto")
        # (M * K) * N overflows UINT64_MAX
        self.assertEqual(simulate_policy(b=2, m=2**32, k=2**32, n=2, threads=4), "serial_auto")
        # matrix_mac * B overflows UINT64_MAX
        self.assertEqual(simulate_policy(b=UINT64_MAX, m=16, k=16, n=16, threads=4), "serial_auto")

    def test_cross_language_parity_golden_vectors(self):
        """
        Verify complete parity against the golden test suite defined in tests/bmm_test.cpp:
        EXPECT_EQ(internal::choose_bmm_execution_policy(...), ...)
        """
        golden_cases = [
            # (B, M, K, N, threads, expected)
            (1, 128, 128, 128, 8, "serial_auto"),
            (0, 128, 128, 128, 8, "serial_auto"),
            (8, 128, 128, 128, 1, "serial_auto"),
            (8, 128, 128, 128, 0, "serial_auto"),
            (2, 4, 4, 4, 4, "serial_auto"),
            (2, 16, 16, 16, 4, "serial_auto"),
            (4, 16, 16, 16, 4, "serial_auto"),
            (4, 20, 20, 20, 4, "serial_auto"),
            (8, 16, 16, 16, 4, "parallel_batch_serial_inner"),
            (8, 20, 20, 20, 4, "parallel_batch_serial_inner"),
            (2, UINT64_MAX, 2, 2, 4, "serial_auto"),
            (8, 32, 32, 32, 4, "parallel_batch_serial_inner"),
            (16, 64, 64, 64, 6, "parallel_batch_serial_inner"),
        ]

        for b, m, k, n, threads, expected in golden_cases:
            with self.subTest(b=b, m=m, k=k, n=n, threads=threads):
                actual = simulate_policy(b, m, k, n, threads)
                self.assertEqual(
                    actual,
                    expected,
                    f"Parity mismatch for (B={b}, M={m}, K={k}, N={n}, threads={threads}): got {actual}, expected {expected}",
                )

    def test_duplicate_same_file_rejection(self):
        """
        Enforce duplicate contract: identical (B, M, K, N, threads, mode) in the SAME file
        must raise ValueError.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["B", "M", "K", "N", "threads", "mode", "median_ms"])
            writer.writerow([4, 64, 64, 64, 4, "serial_auto", 1.25])
            writer.writerow([4, 64, 64, 64, 4, "serial_auto", 1.20])  # Duplicate!
            temp_path = f.name

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_csvs([temp_path])
            self.assertIn("Duplicate configuration in same file", str(ctx.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_duplicate_cross_file_aggregation(self):
        """
        Enforce cross-file aggregation contract:
        - Median of per-file medians is computed.
        - Provenance (sample_count, source_files, raw_medians) is recorded.
        - Output is order-independent.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f3:

            # File 1: measurement 10.0
            w1 = csv.writer(f1)
            w1.writerow(["B", "M", "K", "N", "threads", "mode", "median_ms"])
            w1.writerow([4, 64, 64, 64, 4, "serial_auto", 10.0])
            w1.writerow([4, 64, 64, 64, 4, "parallel_batch_serial_inner", 5.0])
            path1 = f1.name

            # File 2: measurement 20.0
            w2 = csv.writer(f2)
            w2.writerow(["B", "M", "K", "N", "threads", "mode", "median_ms"])
            w2.writerow([4, 64, 64, 64, 4, "serial_auto", 20.0])
            w2.writerow([4, 64, 64, 64, 4, "parallel_batch_serial_inner", 7.0])
            path2 = f2.name

            # File 3: measurement 30.0
            w3 = csv.writer(f3)
            w3.writerow(["B", "M", "K", "N", "threads", "mode", "median_ms"])
            w3.writerow([4, 64, 64, 64, 4, "serial_auto", 30.0])
            w3.writerow([4, 64, 64, 64, 4, "parallel_batch_serial_inner", 6.0])
            path3 = f3.name

        try:
            # Order 1: f1, f2, f3 -> median of [10.0, 20.0, 30.0] is 20.0
            records1, rows1 = load_and_validate_csvs([path1, path2, path3])
            key = (4, 64, 64, 64, 4)
            self.assertEqual(rows1, 6)
            self.assertEqual(records1[key]["serial_auto"]["median_ms"], 20.0)
            self.assertEqual(records1[key]["serial_auto"]["sample_count"], 3)
            self.assertEqual(len(records1[key]["serial_auto"]["source_files"]), 3)

            # Order 2: f3, f1, f2 -> must produce identical aggregated median
            records2, rows2 = load_and_validate_csvs([path3, path1, path2])
            self.assertEqual(records2[key]["serial_auto"]["median_ms"], 20.0)
            self.assertEqual(records2[key]["parallel_batch_serial_inner"]["median_ms"], 6.0)

            # Analyze configurations
            res = analyze_configurations(records1)
            self.assertEqual(res["total_valid_configurations"], 1)
            self.assertEqual(res["win_rates"]["parallel_batch_wins"], 1)
        finally:
            for p in (path1, path2, path3):
                if os.path.exists(p):
                    os.remove(p)

    def test_malformed_csv_negative(self):
        """
        Verify robust rejection of malformed CSVs:
        - Missing columns
        - Non-numeric fields
        - Non-positive or non-finite timings
        """
        # Missing columns
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("B,M,K,mode,median_ms\n4,16,16,serial_auto,1.0\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_and_validate_csvs([path])
        finally:
            os.remove(path)

        # Non-numeric value
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("B,M,K,N,threads,mode,median_ms\n4,16,abc,16,4,serial_auto,1.0\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_and_validate_csvs([path])
        finally:
            os.remove(path)

        # Zero timing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("B,M,K,N,threads,mode,median_ms\n4,16,16,16,4,serial_auto,0.0\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_and_validate_csvs([path])
        finally:
            os.remove(path)

        # Negative timing
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("B,M,K,N,threads,mode,median_ms\n4,16,16,16,4,serial_auto,-0.5\n")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_and_validate_csvs([path])
        finally:
            os.remove(path)

    def test_missing_modes_negative(self):
        """
        Verify that analyze_configurations raises ValueError if no paired modes exist.
        """
        records = {
            (4, 16, 16, 16, 4): {
                "serial_auto": {"median_ms": 1.0, "sample_count": 1, "source_files": ["f.csv"], "raw_medians": [1.0]}
            }
        }
        with self.assertRaises(ValueError):
            analyze_configurations(records)

    def test_cli_format_outputs(self):
        """
        Verify text and markdown report generators render expected sections.
        """
        records = {
            (4, 64, 64, 64, 4): {
                "serial_auto": {"median_ms": 10.0, "sample_count": 1, "source_files": ["f.csv"], "raw_medians": [10.0]},
                "parallel_batch_serial_inner": {"median_ms": 5.0, "sample_count": 1, "source_files": ["f.csv"], "raw_medians": [5.0]},
            }
        }
        analysis = analyze_configurations(records)

        # Text report output
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            print_text_report(analysis)
        finally:
            sys.stdout = old_stdout
        text_output = stdout_capture.getvalue()
        self.assertIn("HELIX BMM Scheduling Policy Analysis Report", text_output)
        self.assertIn("Candidate Policy Parameters", text_output)
        self.assertIn("DISABLED (matching C++ production)", text_output)

        # Markdown report output
        md_capture = io.StringIO()
        try:
            sys.stdout = md_capture
            print_markdown_report(analysis)
        finally:
            sys.stdout = old_stdout
        md_output = md_capture.getvalue()
        self.assertIn("# HELIX BMM Scheduling Policy Analysis Report", md_output)
        self.assertIn("## Win-Rate vs SerialAuto", md_output)
        self.assertIn("## Candidate Policy Regret vs Oracle", md_output)
        self.assertIn("aggregated_median_ms", md_output)


if __name__ == "__main__":
    unittest.main()
