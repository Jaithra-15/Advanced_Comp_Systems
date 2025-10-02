#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Config ----------
INPUT_CSV = "data/mlc_latency_rw.csv"
OUT_DIR = "mlc_combined_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Cache level boundaries in KiB
CACHE_LEVELS = {
    "L1": 32,
    "L2": 256,
    "LLC": 4000,      # 24 MiB
    "DRAM": 52800,   # 128 MiB
}

# ---------- Load ----------
df = pd.read_csv(INPUT_CSV)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# ---------- Helper functions ----------
def process_group(values):
    """Remove top 2 outliers, return mean and std."""
    if len(values) > 2:
        values = np.sort(values)[:-2]  # drop largest 2
    return values.mean(), values.std()

# ---------- Aggregate ----------
agg_records = []

for (rw_mix, size), group in df.groupby(["rw_mix", "size"]):
    lat_mean, lat_std = process_group(group["latency_ns"].values)
    bw_mean, bw_std = process_group(group["bandwidth_mbps"].values)
    agg_records.append({
        "rw_mix": rw_mix,
        "size": size,
        "latency_mean": lat_mean,
        "latency_std": lat_std,
        "bandwidth_mean": bw_mean,
        "bandwidth_std": bw_std,
    })

agg = pd.DataFrame(agg_records)


# ---------- Combined plots across rw_mix ----------

colors = plt.cm.tab10.colors  # color palette

def add_cache_lines(ax):
    """Draw cache level dividers with labels on the vertical line."""
    ylim = ax.get_ylim()
    for i, (label, pos) in enumerate(CACHE_LEVELS.items()):
        ax.axvline(pos, linestyle="--", color="black", alpha=0.5)
        ax.text(pos, (ylim[0] + ylim[1]) / 2,
                label, rotation=90, va="center", ha="right",
                backgroundcolor="white", fontsize=8)

# (1) Latency combined
plt.figure()
for i, (rw_mix, group) in enumerate(agg.groupby("rw_mix")):
    group = group.sort_values("size")
    plt.plot(group["size"], group["latency_mean"], marker="o",
             label=rw_mix, color=colors[i % len(colors)])
    # error bars
    for x, y, s in zip(group["size"], group["latency_mean"], group["latency_std"]):
        plt.vlines(x, y - s, y + s, colors=colors[i % len(colors)], alpha=0.5)

plt.xscale("log")
plt.xlabel("Size (KiB)")
plt.ylabel("Latency (ns)")
plt.title("Latency vs Size (All rw_mix)")
ax = plt.gca()
add_cache_lines(ax)
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_latency.png"))
plt.close()

# (2) Bandwidth combined
plt.figure()
for i, (rw_mix, group) in enumerate(agg.groupby("rw_mix")):
    group = group.sort_values("size")
    plt.plot(group["size"], group["bandwidth_mean"], marker="o",
             label=rw_mix, color=colors[i % len(colors)])
    for x, y, s in zip(group["size"], group["bandwidth_mean"], group["bandwidth_std"]):
        plt.vlines(x, y - s, y + s, colors=colors[i % len(colors)], alpha=0.5)

plt.xscale("log")
plt.xlabel("Size (KiB)")
plt.ylabel("Bandwidth (MB/s)")
plt.title("Bandwidth vs Size (All rw_mix)")
ax = plt.gca()
add_cache_lines(ax)
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_bandwidth.png"))
plt.close()

# (3) Throughput gap combined
plt.figure()
for i, (rw_mix, group) in enumerate(agg.groupby("rw_mix")):
    group = group.sort_values("size")
    max_bw = group["bandwidth_mean"].max()
    gap = max_bw - group["bandwidth_mean"]
    plt.plot(group["size"], gap, marker="o", label=rw_mix, color=colors[i % len(colors)])

plt.xscale("log")
plt.xlabel("Size (KiB)")
plt.ylabel("Throughput Gap (MB/s)")
plt.title("Throughput Gap vs Size (All rw_mix)")
ax = plt.gca()
add_cache_lines(ax)
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "combined_throughput_gap.png"))
plt.close()

plt.figure()

# Filter only r50w50
group = agg[agg["rw_mix"] == "r50w50"].sort_values("bandwidth_mean")

# Compute gap from peak for this rw_mix
max_bw = group["bandwidth_mean"].max()
gap = max_bw - group["bandwidth_mean"]

# Find knee using elbow method
x = gap.values
y = group["latency_mean"].values

# Normalize for curvature detection
x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
coords = np.vstack((x_norm, y_norm)).T

line = np.linspace(coords[0], coords[-1], len(coords))
dist = np.linalg.norm(coords - line, axis=1)
knee_idx = dist.argmax()
knee_x, knee_y = x[knee_idx], y[knee_idx]

# Plot curve
plt.plot(x, y, marker="o", label="", color="C0")

# Error bars for latency
for xi, yi, si in zip(x, y, group["latency_std"]):
    plt.vlines(xi, yi - si, yi + si, colors="C0", alpha=0.5)

# Mark knee
plt.scatter(knee_x, knee_y, color="red", zorder=5)
plt.annotate(f"Knee\n({knee_x:.1f} MB/s gap, {knee_y:.2f} ns)",
             (knee_x, knee_y),
             xytext=(20, 20), textcoords="offset points",
             arrowprops=dict(arrowstyle="->", color="red"))

plt.xlabel("Throughput Gap (MB/s)")
plt.ylabel("Latency (ns)")
plt.title("Intensity Sweep")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "intensity_knee.png"))
plt.close()


print(f"Combined plots saved to {OUT_DIR}")
