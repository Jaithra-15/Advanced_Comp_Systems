# plot_feature3.py
# Better plots for Project A1 Feature 3 (SMT interference).
#
# Tailored to your actual CSV layout:
#   results/feature3_smt.csv
#
# Header (from your example):
#   scenario,thread,run,iters,runtime_seconds
#
# Row pattern (what's really in the file):
#   scenario,thread,ITERS_VALUE,,RUNTIME_SECONDS
#
# We interpret:
#   scenario = row[0]
#   thread   = row[1]
#   iters    = row[2]  (big integer like 1704000000)
#   runtime  = row[-1] (≈ 5.0 seconds)
#
# Outputs:
#   feature3_smt_iters.png      - mean iterations ± 1 std dev
#   feature3_smt_norm.png       - throughput normalized to S1_single_1 = 1.0
#
# Uses a non-GUI backend, so it works fine on WSL/headless without Qt.

import matplotlib
matplotlib.use("Agg")

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def stats(values):
    """Return (mean, std_dev) for a list of floats."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    return mean, std


def load_feature3(csv_path: str):
    """
    Load Feature 3 data with positional parsing.

    Config key = f"{scenario}_{thread}"

    Returns:
        data: dict[config] = {"iters": [..], "time": [..]}
    """
    data = defaultdict(lambda: {"iters": [], "time": []})

    if not os.path.exists(csv_path):
        print(f"[F3] CSV not found: {csv_path}")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[F3] Empty CSV.")
        return {}

    # Skip header row (first line)
    for row in rows[1:]:
        if not row:
            continue
        # Need at least 3 columns: scenario, thread, iters
        if len(row) < 3:
            continue

        scenario = row[0].strip()
        thread = row[1].strip()
        if not scenario or scenario.lower() == "scenario":
            continue

        cfg = f"{scenario}_{thread}"

        # Parse iterations from row[2]
        iters_val = None
        try:
            txt = row[2].strip()
            if txt != "":
                iters_val = float(txt)
        except (IndexError, ValueError):
            iters_val = None

        # Parse runtime from last column
        runtime_val = None
        try:
            last = row[-1].strip()
            if last != "":
                runtime_val = float(last)
        except (IndexError, ValueError):
            runtime_val = None

        if iters_val is not None:
            data[cfg]["iters"].append(iters_val)
        if runtime_val is not None:
            data[cfg]["time"].append(runtime_val)

    if not data:
        print("[F3] No usable rows after parsing.")
    else:
        print(f"[F3] Loaded data for {len(data)} configurations.")

    return data


def plot_feature3_iters(data, out_path: str):
    """
    Plot mean iterations per config (scenario/thread) with ±1 std dev.
    """
    configs = sorted(data.keys())
    mean_iters = []
    std_iters = []
    valid_cfgs = []

    for cfg in configs:
        it_list = data[cfg]["iters"]
        if not it_list:
            continue
        m, s = stats(it_list)
        mean_iters.append(m)
        std_iters.append(s)
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[F3] No iteration data; cannot make iterations plot.")
        return False

    plt.figure()
    plt.bar(valid_cfgs, mean_iters, yerr=std_iters, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scenario / Thread")
    plt.ylabel("Average iterations in 5 s (higher is better)")
    plt.title("Feature 3: SMT Interference\nMean iterations ± 1 · std dev")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F3] Saved iterations plot to {out_path}")
    return True


def plot_feature3_normalized(data, out_path: str, baseline_key: str = "S1_single_1"):
    """
    Plot throughput normalized to a baseline config.

    normalized(cfg) = mean_iters(cfg) / mean_iters(baseline_key)

    If baseline is missing, pick the config with the highest mean_iters as baseline.
    """
    configs = sorted(data.keys())
    mean_iters = {}

    for cfg in configs:
        it_list = data[cfg]["iters"]
        if not it_list:
            continue
        m, _ = stats(it_list)
        if m > 0:
            mean_iters[cfg] = m

    if not mean_iters:
        print("[F3] No iteration data; cannot make normalized plot.")
        return False

    # Choose baseline
    if baseline_key in mean_iters:
        baseline = baseline_key
    else:
        # Fallback: pick config with highest mean_iters as "best" baseline
        baseline = max(mean_iters, key=mean_iters.get)
        print(f"[F3] Baseline '{baseline_key}' not found; using '{baseline}' as baseline.")

    baseline_val = mean_iters[baseline]

    norm_cfgs = sorted(mean_iters.keys())
    norm_vals = [mean_iters[cfg] / baseline_val for cfg in norm_cfgs]

    plt.figure()
    plt.bar(norm_cfgs, norm_vals)
    plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scenario / Thread")
    plt.ylabel(f"Throughput relative to {baseline}")
    plt.title("Feature 3: SMT Interference\nNormalized iterations (baseline = 1.0)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F3] Saved normalized throughput plot to {out_path}")
    return True


if __name__ == "__main__":
    csv_path = "results/feature3_smt.csv"
    data = load_feature3(csv_path)
    if not data:
        raise SystemExit(1)

    # Plot 1: raw iterations (mean ± 1 std)
    plot_feature3_iters(data, "feature3_smt_iters.png")

    # Plot 2: normalized throughput vs baseline S1_single_1
    plot_feature3_normalized(data, "feature3_smt_norm.png", baseline_key="S1_single_1")

