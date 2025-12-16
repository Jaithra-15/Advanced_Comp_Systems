# plots_f13.py
# Plot Feature 1 and Feature 3 for Project A1.
#
# This script is tailored to the actual CSVs you showed:
#   Feature 1 (results/feature1_affinity.csv):
#     header: config,run,iters,runtime_seconds
#     but rows look like: config,run,iters,,runtime
#     => runtime is actually in the LAST column.
#
#   Feature 3 (results/feature3_smt.csv):
#     header: scenario,thread,run,iters,runtime_seconds
#     rows: scenario,thread,run,iters?,runtime
#
# It will:
#   - Plot iterations (mean ± 1 std) for F1 if possible.
#   - Also plot estimated time to reach X iterations for F1.
#   - For F3, plot iterations if available; otherwise plot runtime.
#
# It uses a non-GUI backend, so no Qt/Wayland issues.

import matplotlib
matplotlib.use("Agg")  # render directly to files

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


# ======================== Feature 1 ========================

def load_feature1(csv_path):
    """
    Load Feature 1 data using csv.reader (to handle extra comma).

    Expected pattern (from your example):
        header: config,run,iters,runtime_seconds
        row:   config,run,iters,,runtime

    We treat:
        config = row[0]
        run    = row[1]
        iters  = row[2]
        runtime_seconds = row[-1]  (always last column)
    """
    data = defaultdict(lambda: {"iters": [], "time": []})

    if not os.path.exists(csv_path):
        print(f"[F1] CSV not found: {csv_path}")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[F1] Empty CSV.")
        return {}

    # Skip header row
    for row in rows[1:]:
        if not row:
            continue

        # Must at least have config, run, iters
        if len(row) < 3:
            continue

        config = row[0].strip()
        if config == "" or config.lower() == "config":
            continue

        # iters column
        iters_val = None
        try:
            if row[2].strip() != "":
                iters_val = float(row[2].strip())
        except ValueError:
            iters_val = None

        # runtime is always the last column (handles extra comma)
        runtime_val = None
        try:
            last = row[-1].strip()
            if last != "":
                runtime_val = float(last)
        except ValueError:
            runtime_val = None

        if runtime_val is not None:
            data[config]["time"].append(runtime_val)
        if iters_val is not None:
            data[config]["iters"].append(iters_val)

    if not data:
        print("[F1] No usable rows after parsing.")
    return data


def plot_feature1_iters(data, out_path):
    """
    Plot mean iterations per config for Feature 1 with ±1 std dev.
    If no iterations exist, returns False (caller can fall back to runtime).
    """
    configs = sorted(data.keys())
    mean_iters = []
    std_iters = []
    valid_cfgs = []

    for cfg in configs:
        if not data[cfg]["iters"]:
            continue
        m, s = stats(data[cfg]["iters"])
        mean_iters.append(m)
        std_iters.append(s)
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[F1] No iteration data; skipping F1 iterations plot.")
        return False

    plt.figure()
    plt.bar(valid_cfgs, mean_iters, yerr=std_iters, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel("Average iterations in 5 s (higher is better)")
    plt.title("Feature 1: CPU Affinity\nMean iterations ± 1 · std dev")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F1] Saved iterations plot to {out_path}")
    return True


def plot_feature1_runtime(data, out_path):
    """Fallback: plot mean runtime per config with ±1 std dev."""
    configs = sorted(data.keys())
    mean_time = []
    std_time = []
    valid_cfgs = []

    for cfg in configs:
        if not data[cfg]["time"]:
            continue
        m, s = stats(data[cfg]["time"])
        mean_time.append(m)
        std_time.append(s)
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[F1] No runtime data; cannot plot F1.")
        return False

    plt.figure()
    plt.bar(valid_cfgs, mean_time, yerr=std_time, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel("Average runtime (s)")
    plt.title("Feature 1: CPU Affinity\nMean runtime ± 1 · std dev")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F1] Saved runtime plot to {out_path}")
    return True


def plot_feature1_time_to_X(data, out_path):
    """
    Plot estimated time to reach a common iteration target X for F1.

    X = min(mean_iters over configs with iteration data).
    t_X(cfg) = mean_time(cfg) * X / mean_iters(cfg).
    """
    configs = sorted(data.keys())
    mean_iters = {}
    mean_time = {}

    # Build mean iters + mean time where both exist
    for cfg in configs:
        if not data[cfg]["iters"] or not data[cfg]["time"]:
            continue
        mi, _ = stats(data[cfg]["iters"])
        mt, _ = stats(data[cfg]["time"])
        if mi > 0 and mt > 0:
            mean_iters[cfg] = mi
            mean_time[cfg] = mt

    if not mean_iters:
        print("[F1] No configs with both iters and time; skipping time-to-X plot.")
        return

    X = min(mean_iters.values())
    cfgs_used = sorted(mean_iters.keys())
    est_times = []

    for cfg in cfgs_used:
        mi = mean_iters[cfg]
        mt = mean_time[cfg]
        est_times.append(mt * (X / mi))

    plt.figure()
    plt.bar(cfgs_used, est_times)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel(f"Estimated time to reach {int(X)} iterations (s)")
    plt.title("Feature 1: CPU Affinity\nEstimated time to common work X")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F1] Saved time-to-X plot to {out_path}")


def do_feature1():
    csv_path = "results/feature1_affinity.csv"
    data = load_feature1(csv_path)
    if not data:
        return

    # Try iterations plot first
    ok = plot_feature1_iters(data, "feature1_affinity_iters.png")
    if not ok:
        # Fallback to runtime-only plot
        plot_feature1_runtime(data, "feature1_affinity_iters.png")

    # Only attempt time-to-X if iterations exist
    if any(len(v["iters"]) > 0 for v in data.values()):
        plot_feature1_time_to_X(data, "feature1_affinity_time_to_X.png")


# ======================== Feature 3 ========================

def load_feature3(csv_path):
    """
    Load Feature 3 data using csv.reader.

    Expected pattern:
        header: scenario,thread,run,iters,runtime_seconds
        row:   scenario,thread,run,iters?,runtime

    We treat config = f"{scenario}_{thread}".
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

    for row in rows[1:]:
        if not row or len(row) < 4:
            continue

        scenario = row[0].strip()
        thread = row[1].strip()
        if scenario.lower() == "scenario":
            continue

        cfg = f"{scenario}_{thread}"

        # iters is row[3]
        iters_val = None
        try:
            if row[3].strip() != "":
                iters_val = float(row[3].strip())
        except ValueError:
            iters_val = None

        # runtime is last column
        runtime_val = None
        try:
            last = row[-1].strip()
            if last != "":
                runtime_val = float(last)
        except ValueError:
            runtime_val = None

        if runtime_val is not None:
            data[cfg]["time"].append(runtime_val)
        if iters_val is not None:
            data[cfg]["iters"].append(iters_val)

    if not data:
        print("[F3] No usable rows after parsing.")
    return data


def plot_feature3_iters_or_time(data, out_path):
    """
    For Feature 3:
      - If iterations exist, plot iterations (mean ± 1 std).
      - Else, plot runtime (mean ± 1 std).
    """
    configs = sorted(data.keys())
    any_iters = any(len(v["iters"]) > 0 for v in data.values())

    if any_iters:
        mean_iters = []
        std_iters = []
        valid_cfgs = []

        for cfg in configs:
            if not data[cfg]["iters"]:
                continue
            m, s = stats(data[cfg]["iters"])
            mean_iters.append(m)
            std_iters.append(s)
            valid_cfgs.append(cfg)

        if valid_cfgs:
            plt.figure()
            plt.bar(valid_cfgs, mean_iters, yerr=std_iters, capsize=5)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Scenario / Thread")
            plt.ylabel("Average iterations in 5 s")
            plt.title("Feature 3: SMT Interference\nMean iterations ± 1 · std dev")
            plt.tight_layout()
            plt.savefig(out_path)
            print(f"[F3] Saved iterations plot to {out_path}")
            return
        else:
            print("[F3] Iteration arrays empty; falling back to runtime plot.")

    # No iterations → runtime plot
    mean_time = []
    std_time = []
    valid_cfgs = []

    for cfg in configs:
        if not data[cfg]["time"]:
            continue
        m, s = stats(data[cfg]["time"])
        mean_time.append(m)
        std_time.append(s)
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[F3] No runtime data; cannot plot F3.")
        return

    plt.figure()
    plt.bar(valid_cfgs, mean_time, yerr=std_time, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scenario / Thread")
    plt.ylabel("Average runtime (s)")
    plt.title("Feature 3: SMT Interference\nMean runtime ± 1 · std dev")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[F3] Saved runtime plot to {out_path}")


def do_feature3():
    csv_path = "results/feature3_smt.csv"
    data = load_feature3(csv_path)
    if not data:
        return
    plot_feature3_iters_or_time(data, "feature3_smt.png")


# ======================== main ========================

if __name__ == "__main__":
    do_feature1()
    do_feature3()

