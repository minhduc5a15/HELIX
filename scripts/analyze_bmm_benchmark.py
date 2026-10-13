#!/usr/bin/env python3
"""
HELIX BMM Scheduling Policy Analyzer
Reproducible analysis of BMM benchmark results, regret metrics, and candidate threshold evaluation.
Uses only Python Standard Library.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze HELIX BMM benchmark CSV results.")
    parser.add_argument("csv_files", nargs="+", type=str, help="Path(s) to benchmark CSV file(s).")
    parser.add_argument(
        "--min-total-mac",
        type=int,
        default=32768,
        help="Candidate minimum total MAC threshold for ParallelBatch (default: 32768).",
    )
    parser.add_argument(
        "--large-matrix-mac",
        type=int,
        default=512 * 512 * 512,
        help="Candidate per-matrix MAC threshold for under-utilization guard (default: 512^3 = 134217728).",
    )
    parser.add_argument(
        "--near-tie-tolerance",
        type=float,
        default=0.03,
        help="Tolerance ratio (e.g. 0.03 for 3%%) to consider two modes tied (default: 0.03).",
    )
    parser.add_argument("--output", type=str, default="", help="Path to export JSON analysis report.")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "both"],
        default="text",
        help="Format for console summary output (default: text).",
    )
    return parser.parse_args()


def load_and_validate_csvs(filepaths):
    required_cols = {"B", "M", "K", "N", "threads", "mode", "median_ms"}
    records_by_config = defaultdict(dict)
    total_rows = 0

    for path_str in filepaths:
        p = Path(path_str)
        if not p.is_file():
            print(f"Error: CSV file not found: {path_str}", file=sys.stderr)
            sys.exit(1)

        with open(p, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                print(f"Error: {path_str} is missing required columns: {missing}", file=sys.stderr)
                sys.exit(1)

            for line_no, row in enumerate(reader, start=2):
                total_rows += 1
                try:
                    b = int(row["B"])
                    m = int(row["M"])
                    k = int(row["K"])
                    n = int(row["N"])
                    threads = int(row["threads"])
                    mode = row["mode"].strip()
                    median_ms = float(row["median_ms"])
                except (ValueError, TypeError) as e:
                    print(f"Error: Invalid data in {path_str}:{line_no}: {e}", file=sys.stderr)
                    sys.exit(1)

                if median_ms <= 0 or not math.isfinite(median_ms):
                    print(
                        f"Error: Non-positive or non-finite timing in {path_str}:{line_no}: {median_ms}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                key = (b, m, k, n, threads)
                records_by_config[key][mode] = median_ms

    return records_by_config, total_rows


def simulate_policy(b, m, k, n, threads, min_total_mac, large_matrix_mac):
    if b <= 1 or threads <= 1:
        return "serial_auto"

    matrix_mac = m * k * n
    total_mac = b * matrix_mac

    if total_mac < min_total_mac:
        return "serial_auto"

    if b < threads and matrix_mac >= large_matrix_mac:
        return "serial_auto"

    return "parallel_batch_serial_inner"


def analyze_configurations(records_by_config, min_total_mac, large_matrix_mac, tie_tol):
    paired = []
    missing_paired = 0

    for (b, m, k, n, threads), modes in records_by_config.items():
        if "serial_auto" in modes and "parallel_batch_serial_inner" in modes:
            s_time = modes["serial_auto"]
            p_time = modes["parallel_batch_serial_inner"]
            matrix_mac = m * k * n
            total_mac = b * matrix_mac
            paired.append({
                "B": b,
                "M": m,
                "K": k,
                "N": n,
                "threads": threads,
                "matrix_mac": matrix_mac,
                "total_mac": total_mac,
                "serial_auto_ms": s_time,
                "parallel_batch_ms": p_time,
            })
        else:
            missing_paired += 1

    if not paired:
        print("Error: No valid paired configurations (serial_auto vs parallel_batch_serial_inner) found.", file=sys.stderr)
        sys.exit(1)

    # 1. Win-Rate Analysis
    parallel_wins = 0
    serial_wins = 0
    ties = 0
    speedups = []

    for item in paired:
        s = item["serial_auto_ms"]
        p = item["parallel_batch_ms"]
        speedup = s / p
        speedups.append(speedup)

        ratio = abs(s - p) / min(s, p)
        if ratio <= tie_tol:
            ties += 1
        elif p < s:
            parallel_wins += 1
        else:
            serial_wins += 1

    total_valid = len(paired)
    speedups.sort()
    p50_speedup = statistics.median(speedups)
    p90_speedup = speedups[int(total_valid * 0.90)] if total_valid >= 10 else speedups[-1]
    p95_speedup = speedups[int(total_valid * 0.95)] if total_valid >= 20 else speedups[-1]

    # 2. Policy Regret Analysis
    regrets = []
    regrets_by_thread = defaultdict(list)
    regrets_by_work = defaultdict(list)  # small (<32k), mid (<512k), large
    policy_misses = []

    for item in paired:
        b = item["B"]
        m = item["M"]
        k = item["K"]
        n = item["N"]
        threads = item["threads"]
        s_time = item["serial_auto_ms"]
        p_time = item["parallel_batch_ms"]

        oracle_ms = min(s_time, p_time)
        chosen_mode = simulate_policy(b, m, k, n, threads, min_total_mac, large_matrix_mac)
        chosen_ms = s_time if chosen_mode == "serial_auto" else p_time

        regret = (chosen_ms / oracle_ms) - 1.0
        regrets.append(regret)
        regrets_by_thread[threads].append(regret)

        total_mac = item["total_mac"]
        if total_mac < 32768:
            work_bucket = "< 32K MACs"
        elif total_mac < 524288:
            work_bucket = "32K - 512K MACs"
        else:
            work_bucket = ">= 512K MACs"
        regrets_by_work[work_bucket].append(regret)

        if regret > 0.03:
            policy_misses.append({
                "B": b,
                "M": m,
                "K": k,
                "N": n,
                "threads": threads,
                "total_mac": total_mac,
                "chosen": chosen_mode,
                "oracle": "serial_auto" if s_time < p_time else "parallel_batch_serial_inner",
                "chosen_ms": chosen_ms,
                "oracle_ms": oracle_ms,
                "regret_percent": regret * 100.0,
            })

    regrets.sort()
    med_regret = statistics.median(regrets)
    p95_regret = regrets[int(total_valid * 0.95)] if total_valid >= 20 else regrets[-1]
    max_regret = max(regrets)
    regret_gt_5 = sum(1 for r in regrets if r > 0.05) / total_valid * 100.0
    regret_gt_10 = sum(1 for r in regrets if r > 0.10) / total_valid * 100.0

    policy_misses.sort(key=lambda x: x["regret_percent"], reverse=True)

    thread_summary = {}
    for t, t_regrets in sorted(regrets_by_thread.items()):
        t_regrets.sort()
        thread_summary[t] = {
            "count": len(t_regrets),
            "median_regret": statistics.median(t_regrets) * 100.0,
            "p95_regret": (t_regrets[int(len(t_regrets) * 0.95)] if len(t_regrets) >= 20 else t_regrets[-1]) * 100.0,
            "max_regret": max(t_regrets) * 100.0,
        }

    work_summary = {}
    for wb, wb_regrets in sorted(regrets_by_work.items()):
        wb_regrets.sort()
        work_summary[wb] = {
            "count": len(wb_regrets),
            "median_regret": statistics.median(wb_regrets) * 100.0,
            "p95_regret": (wb_regrets[int(len(wb_regrets) * 0.95)] if len(wb_regrets) >= 20 else wb_regrets[-1]) * 100.0,
            "max_regret": max(wb_regrets) * 100.0,
        }

    return {
        "total_valid_configurations": total_valid,
        "missing_paired_configurations": missing_paired,
        "candidate_policy": {
            "min_total_mac": min_total_mac,
            "large_matrix_mac": large_matrix_mac,
        },
        "win_rates": {
            "parallel_batch_wins": parallel_wins,
            "parallel_batch_win_rate": parallel_wins / total_valid * 100.0,
            "serial_auto_wins": serial_wins,
            "serial_auto_win_rate": serial_wins / total_valid * 100.0,
            "ties": ties,
            "tie_rate": ties / total_valid * 100.0,
            "median_speedup": p50_speedup,
            "p90_speedup": p90_speedup,
            "p95_speedup": p95_speedup,
        },
        "policy_regret": {
            "median_regret_percent": med_regret * 100.0,
            "p95_regret_percent": p95_regret * 100.0,
            "max_regret_percent": max_regret * 100.0,
            "regret_gt_5_percent": regret_gt_5,
            "regret_gt_10_percent": regret_gt_10,
        },
        "by_thread": thread_summary,
        "by_work_volume": work_summary,
        "top_misses": policy_misses[:10],
    }


def print_text_report(res):
    cp = res["candidate_policy"]
    wr = res["win_rates"]
    pr = res["policy_regret"]

    print("=" * 78)
    print(" HELIX BMM Scheduling Policy Analysis Report")
    print("=" * 78)
    print(f"Candidate Policy Parameters:")
    print(f"  Minimum Total MAC Threshold : {cp['min_total_mac']:,} MACs (~{cp['min_total_mac']*2:,} FLOPs)")
    print(f"  Large-Matrix MAC Guard      : {cp['large_matrix_mac']:,} MACs (when B < threads)")
    print(f"Evaluated Configurations      : {res['total_valid_configurations']} paired cases")
    print("-" * 78)
    print(f"Pairwise Win-Rate vs SerialAuto:")
    print(f"  ParallelBatch Wins          : {wr['parallel_batch_wins']:>4} ({wr['parallel_batch_win_rate']:5.1f}%)")
    print(f"  SerialAuto Wins             : {wr['serial_auto_wins']:>4} ({wr['serial_auto_win_rate']:5.1f}%)")
    print(f"  Near-Ties (within 3%)       : {wr['ties']:>4} ({wr['tie_rate']:5.1f}%)")
    print(f"  Median Speedup (PBatch/Auto): {wr['median_speedup']:5.2f}x (p95: {wr['p95_speedup']:5.2f}x)")
    print("-" * 78)
    print(f"Candidate Policy Regret vs Oracle:")
    print(f"  Median Regret               : {pr['median_regret_percent']:5.2f}%")
    print(f"  p95 Regret                  : {pr['p95_regret_percent']:5.2f}%")
    print(f"  Max Regret (Worst Case)     : {pr['max_regret_percent']:5.2f}%")
    print(f"  Cases with Regret > 5%      : {pr['regret_gt_5_percent']:5.1f}%")
    print(f"  Cases with Regret > 10%     : {pr['regret_gt_10_percent']:5.1f}%")
    print("-" * 78)
    print("Regret by Thread Count:")
    print(f"  {'Threads':>7} | {'Cases':>5} | {'Median Regret':>14} | {'p95 Regret':>11} | {'Max Regret':>11}")
    print("  " + "-" * 57)
    for t, data in res["by_thread"].items():
        print(f"  {t:>7} | {data['count']:>5} | {data['median_regret']:13.2f}% | {data['p95_regret']:10.2f}% | {data['max_regret']:10.2f}%")
    print("-" * 78)
    print("Regret by Workload Volume:")
    print(f"  {'Volume Bucket':>18} | {'Cases':>5} | {'Median Regret':>14} | {'p95 Regret':>11} | {'Max Regret':>11}")
    print("  " + "-" * 68)
    for wb, data in res["by_work_volume"].items():
        print(f"  {wb:>18} | {data['count']:>5} | {data['median_regret']:13.2f}% | {data['p95_regret']:10.2f}% | {data['max_regret']:10.2f}%")
    print("-" * 78)

    if res["top_misses"]:
        print("Top Policy Misses (Regret > 3%):")
        print(f"  {'Shape [B,MxKxN]':<20} {'T':>2} {'Total MACs':>11} {'Chosen':<14} {'Oracle':<14} {'Regret':>8}")
        print("  " + "-" * 73)
        for m in res["top_misses"][:8]:
            shape_str = f"[{m['B']}, {m['M']}x{m['K']}x{m['N']}]"
            print(f"  {shape_str:<20} {m['threads']:>2} {m['total_mac']:>11,} {m['chosen']:<14} {m['oracle']:<14} {m['regret_percent']:7.2f}%")
    else:
        print("Top Policy Misses: None (all decisions within 3% of oracle).")
    print("=" * 78)


def main():
    args = parse_args()
    records_by_config, _ = load_and_validate_csvs(args.csv_files)
    analysis = analyze_configurations(records_by_config, args.min_total_mac, args.large_matrix_mac, args.near_tie_tolerance)

    if args.format in ("text", "both"):
        print_text_report(analysis)

    if args.output:
        out_p = Path(args.output)
        if out_p.parent:
            out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis JSON exported to: {args.output}")


if __name__ == "__main__":
    main()
