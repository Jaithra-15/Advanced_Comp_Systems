#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIG ---
CPU_FREQ = 3.2e9  # Hz, adjust this to your fixed CPU frequency
BYTES_PER_FLOP = 6  # for SAXPY: 12B per 2 FLOPs = 6B per flop

# --- LOAD DATA ---
df = pd.read_csv("full_sweep_simd.csv")

# --- FILTER DATA ---
df = df[
    (df["kernel"] == "saxpy") &
    (df["dtype"] == "f32") &
    (df["misaligned"] == False) &
    (df["tail_multiple"] == True)
]

# --- COMPUTE METRICS ---
df["latency_ns"] = df["CPE"] / CPU_FREQ * 1e9
df["bandwidth_MBps"] = df["GFLOPs"] * BYTES_PER_FLOP / 1e3  # MB/s

# --- AGGREGATE OVER TRIALS ---
agg = df.groupby("N", as_index=False).agg(
    latency_ns=("latency_ns", "mean"),
    bandwidth_MBps=("bandwidth_MBps", "mean"),
    latency_std=("latency_ns", "std"),
    bandwidth_std=("bandwidth_MBps", "std")
)

print("Aggregated results:")
print(agg.head())

# --- PLOT BANDWIDTH VS LATENCY ---
plt.figure(figsize=(8,6))
plt.errorbar(
    agg["latency_ns"], agg["bandwidth_MBps"],
    xerr=agg["latency_std"], yerr=agg["bandwidth_std"],
    fmt="o-", capsize=4, label="saxpy f32"
)
plt.xlabel("Latency (ns)")
plt.ylabel("Bandwidth (MB/s)")
plt.title("SAXPY f32 Intensity Sweep (misaligned=False, tail_multiple=True)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("saxpy_f32_intensity.png", dpi=200)
plt.show()
