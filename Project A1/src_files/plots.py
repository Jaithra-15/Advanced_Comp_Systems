# plots.py
# Plot results for Project A1 (Features 1–4).
#
# Uses a non-GUI backend (Agg), so it works on WSL/headless systems
# and does not require Qt or Wayland.
#
# Expected CSVs (from run_experiments.sh):
#
#  Feature 1: results/feature1_affinity.csv
#    columns: config,run,iters,runtime_seconds
#
#  Feature 2: results/feature2_thp.csv
#    columns: pattern,thp_mode,run,runtime_seconds
#
#  Feature 3: results/feature3_smt.csv
#    columns: scenario,thread,run,iters,runtime_seconds
#
#  Feature 4: results/feature4_prefetch.csv
#    columns: pattern,stride,run,runtime_seconds
#
# Plots:
#   feature1_affinity_iters.png
#   feature1_affinity_time_to_X.png
#   feature2_thp_runtime.png
#   feature3_smt_iters.png
#   feature4_prefetch_runtime.png

import matplotlib
matplotlib.use("Agg")  # Avoid Qt/Wayland; render directly to files

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


# ---------------------- common helpers ---------------------- #

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


# ====================== Feature 1 ====================== #

def load_feature1(csv_path):
    """
    Load Feature 1 (affinity) data.

    Expected columns:
        config,run,iters,runtime_seconds
    """
    data = defaultdict(lambda: {"iters": [], "time": []})

    if not os.path.exists(csv_path):
        print(f"[Feature 1] CSV not found: {csv_path} (skipping Feature 1 plots)")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        has_iters = "iters" in fields
        has_time = "runtime_seconds" in fields

        if not has_time:
            print("[Feature 1] CSV missing 'runtime_seconds'; skipping Feature 1.")
            return {}

        if not has_iters:
            print("[Feature 1] CSV missing 'iters'; iteration-based plots will be skipped.")

        for row in reader:
            cfg = row.get("config", "").strip()
            if not cfg:
                continue

            # time is required
            try:
                t = float(row["runtime_seconds"])
            except (KeyError, ValueError):
                continue
            data[cfg]["time"].append(t)

            # iters is optional
            if has_iters:
                try:
                    it = float(row["iters"])
                except (KeyError, ValueError):
                    # bad iters for this row: just skip iterations, keep time
                    continue
                data[cfg]["iters"].append(it)

    return data


def plot_feature1_iters_or_time(data, out_path):
    """
    Plot mean iterations per config with +/- 1 std dev if iteration data exists.
    If no iterations are available, fall back to plotting runtime instead
    (so the graph is never empty).
    """
    configs = sorted(data.keys())
    any_iters = any(len(v["iters"]) > 0 for v in data.values())

    if any_iters:
        # Try iterations-based plot
        mean_vals = []
        std_vals = []
        valid_cfgs = []

        for cfg in configs:
            if not data[cfg]["iters"]:
                continue
            m, s = stats(data[cfg]["iters"])
            mean_vals.append(m)
            std_vals.append(s)  # ±1 std dev
            valid_cfgs.append(cfg)

        if valid_cfgs:
            plt.figure()
            plt.bar(valid_cfgs, mean_vals, yerr=std_vals, capsize=5)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Configuration")
            plt.ylabel("Average iterations in 5 s (higher is better)")
            plt.title("Feature 1: CPU Affinity\nMean iterations ± 1 · std dev")
            plt.tight_layout()
            plt.savefig(out_path)
            print(f"[Feature 1] Saved iterations plot to {out_path}")
            return
        else:
            print("[Feature 1] Iteration arrays empty after parsing; "
                  "falling back to runtime plot.")

    # Fallback: runtime-based plot
    print("[Feature 1] No usable iteration data; plotting runtime instead.")
    mean_time = []
    std_time = []
    valid_cfgs = []

    for cfg in configs:
        if not data[cfg]["time"]:
            continue
        m, s = stats(data[cfg]["time"])
        mean_time.append(m)
        std_time.append(s)  # ±1 std dev
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[Feature 1] No usable runtime data either; skipping plot.")
        return

    plt.figure()
    plt.bar(valid_cfgs, mean_time, yerr=std_time, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel("Average runtime (s)")
    plt.title("Feature 1: CPU Affinity\nMean runtime ± 1 · std dev (fallback)")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Feature 1] Saved runtime fallback plot to {out_path}")


def plot_feature1_time_to_X(data, out_path):
    """
    Plot estimated time to reach a common iteration target X_iters.

    X_iters = min(mean_iters across configs with valid iteration data).

    t_X(config) = mean_time(config) * (X_iters / mean_iters(config)).
    """
    any_iters = any(len(v["iters"]) > 0 for v in data.values())
    if not any_iters:
        print("[Feature 1] No iteration data; skipping time-to-X plot.")
        return

    configs = sorted(data.keys())
    mean_iters = {}
    mean_time = {}

    for cfg in configs:
        if not data[cfg]["iters"] or not data[cfg]["time"]:
            continue
        mi, _ = stats(data[cfg]["iters"])
        mt, _ = stats(data[cfg]["time"])
        if mi > 0 and mt > 0:
            mean_iters[cfg] = mi
            mean_time[cfg] = mt

    if not mean_iters:
        print("[Feature 1] No configs with valid iters+time; skipping time-to-X plot.")
        return

    X_iters = min(mean_iters.values())
    cfgs_used = sorted(mean_iters.keys())
    est_times = []

    for cfg in cfgs_used:
        mi = mean_iters[cfg]
        mt = mean_time[cfg]
        tX = mt * (X_iters / mi)
        est_times.append(tX)

    plt.figure()
    plt.bar(cfgs_used, est_times)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Configuration")
    plt.ylabel(f"Estimated time to reach {int(X_iters)} iterations (s)")
    plt.title("Feature 1: CPU Affinity\nEstimated time to common work X")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[Feature 1] Saved time-to-X plot to {out_path}")


def do_feature1():
    csv_path = "results/feature1_affinity.csv"
    data = load_feature1(csv_path)
    if not data:
        return
    plot_feature1_iters_or_time(data, "feature1_affinity_iters.png")
    plot_feature1_time_to_X(data, "feature1_affinity_time_to_X.png")


# ====================== Feature 2 ====================== #

def load_feature2(csv_path):
    """
    Load Feature 2 (THP) data.

    Expected columns:
        pattern,thp_mode,run,runtime_seconds
    """
    data = defaultdict(list)  # key: (pattern, thp_mode) -> [runtime]

    if not os.path.exists(csv_path):
        print(f"[Feature 2] CSV not found: {csv_path} (skipping Feature 2 plots)")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"pattern", "thp_mode", "runtime_seconds"}
        if not required.issubset(fields):
            print("[Feature 2] CSV missing required columns; skipping Feature 2.")
            return {}

        for row in reader:
            pattern = row["pattern"].strip()
            thp_mode = row["thp_mode"].strip()
            try:
                t = float(row["runtime_seconds"])
            except (KeyError, ValueError):
                continue
            data[(pattern, thp_mode)].append(t)

    return data


def do_feature2():
    csv_path = "results/feature2_thp.csv"
    data = load_feature2(csv_path)
    if not data:
        return

    patterns = sorted({p for (p, _) in data.keys()})
    modes = ["never", "always"]

    import numpy as np
    x = np.arange(len(patterns))
    width = 0.35

    means = {m: [] for m in modes}
    stds = {m: [] for m in modes}

    for pat in patterns:
        for m in modes:
            vals = data.get((pat, m), [])
            mu, sd = stats(vals)
            means[m].append(mu)
            stds[m].append(sd)  # ±1 std dev

    plt.figure()
    plt.bar(x - width/2, means["never"], width, yerr=stds["never"],
            capsize=5, label="THP=never")
    plt.bar(x + width/2, means["always"], width, yerr=stds["always"],
            capsize=5, label="THP=always")
    plt.xticks(x, patterns)
    plt.xlabel("Access pattern")
    plt.ylabel("Average runtime (s)")
    plt.title("Feature 2: THP impact on runtime\nMean ± 1 · std dev")
    plt.legend()
    plt.tight_layout()
    out_path = "feature2_thp_runtime.png"
    plt.savefig(out_path)
    print(f"[Feature 2] Saved runtime plot to {out_path}")


# ====================== Feature 3 ====================== #

def load_feature3(csv_path):
    """
    Load Feature 3 (SMT) data.

    Expected columns:
        scenario,thread,run,iters,runtime_seconds
    """
    data = defaultdict(lambda: {"iters": [], "time": []})

    if not os.path.exists(csv_path):
        print(f"[Feature 3] CSV not found: {csv_path} (skipping Feature 3 plots)")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"scenario", "runtime_seconds"}
        if not required.issubset(fields):
            print("[Feature 3] CSV missing required columns; skipping Feature 3.")
            return {}

        has_iters = "iters" in fields

        for row in reader:
            scenario = row["scenario"].strip()
            thread = row.get("thread", "").strip()
            cfg = scenario if not thread else f"{scenario}_{thread}"

            try:
                t = float(row["runtime_seconds"])
            except (KeyError, ValueError):
                continue
            data[cfg]["time"].append(t)

            if has_iters:
                try:
                    it = float(row["iters"])
                except (KeyError, ValueError):
                    continue
                data[cfg]["iters"].append(it)

    return data


def do_feature3():
    csv_path = "results/feature3_smt.csv"
    data = load_feature3(csv_path)
    if not data:
        return

    any_iters = any(len(v["iters"]) > 0 for v in data.values())
    if not any_iters:
        print("[Feature 3] No iteration data; SMT plot will use runtime only.")
        # Could add runtime-only plot here if needed.
        return

    configs = sorted(data.keys())
    mean_iters = []
    std_iters = []
    valid_cfgs = []

    for cfg in configs:
        if not data[cfg]["iters"]:
            continue
        m, s = stats(data[cfg]["iters"])
        mean_iters.append(m)
        std_iters.append(s)  # ±1 std dev
        valid_cfgs.append(cfg)

    if not valid_cfgs:
        print("[Feature 3] Iteration arrays empty; skipping Feature 3 plot.")
        return

    plt.figure()
    plt.bar(valid_cfgs, mean_iters, yerr=std_iters, capsize=5)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Scenario / Thread")
    plt.ylabel("Average iterations in 5 s")
    plt.title("Feature 3: SMT Interference\nMean iterations ± 1 · std dev")
    plt.tight_layout()
    out_path = "feature3_smt_iters.png"
    plt.savefig(out_path)
    print(f"[Feature 3] Saved iterations plot to {out_path}")


# ====================== Feature 4 ====================== #

def load_feature4(csv_path):
    """
    Load Feature 4 (prefetch/stride) data.

    Expected columns:
        pattern,stride,run,runtime_seconds
    """
    data = defaultdict(list)  # key: (pattern, stride) -> [time]

    if not os.path.exists(csv_path):
        print(f"[Feature 4] CSV not found: {csv_path} (skipping Feature 4 plots)")
        return {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"pattern", "stride", "runtime_seconds"}
        if not required.issubset(fields):
            print("[Feature 4] CSV missing required columns; skipping Feature 4.")
            return {}

        for row in reader:
            pattern = row["pattern"].strip()
            try:
                stride = int(row["stride"])
                t = float(row["runtime_seconds"])
            except (ValueError, KeyError):
                continue
            key = (pattern, stride)
            data[key].append(t)

    return data


def do_feature4():
    csv_path = "results/feature4_prefetch.csv"
    data = load_feature4(csv_path)
    if not data:
        return

    # Focus on stride pattern as a function of stride, with seq/rand as references.
    strides = sorted({s for (p, s) in data.keys() if p == "stride"})
    mean_stride = []
    std_stride = []

    for s in strides:
        vals = data.get(("stride", s), [])
        m, sd = stats(vals)
        mean_stride.append(m)
        std_stride.append(sd)  # ±1 std dev

    # Optionally get seq and rand (stride field arbitrary there)
    seq_vals = []
    rand_vals = []
    for (p, s), vals in data.items():
        if p == "seq":
            seq_vals.extend(vals)
        elif p == "rand":
            rand_vals.extend(vals)

    m_seq, _ = stats(seq_vals) if seq_vals else (float("nan"), float("nan"))
    m_rand, _ = stats(rand_vals) if rand_vals else (float("nan"), float("nan"))

    import numpy as np
    plt.figure()
    plt.errorbar(strides, mean_stride, yerr=std_stride, fmt='-o', capsize=5,
                 label="stride pattern")

    if not math.isnan(m_seq):
        plt.axhline(m_seq, linestyle="--", label=f"seq (≈{m_seq:.3f} s)")
    if not math.isnan(m_rand):
        plt.axhline(m_rand, linestyle=":", label=f"rand (≈{m_rand:.3f} s)")

    plt.xlabel("Stride (elements)")
    plt.ylabel("Average runtime (s)")
    plt.title("Feature 4: Prefetcher / stride effects\nMean runtime ± 1 · std dev")
    plt.legend()
    plt.tight_layout()
    out_path = "feature4_prefetch_runtime.png"
    plt.savefig(out_path)
    print(f"[Feature 4] Saved runtime plot to {out_path}")


# ====================== main ====================== #

if __name__ == "__main__":
    do_feature1()
    do_feature2()
    do_feature3()
    do_feature4()

