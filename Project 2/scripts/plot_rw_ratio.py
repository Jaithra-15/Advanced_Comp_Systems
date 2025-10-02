#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV
df = pd.read_csv("data/mlc_latency_rw.csv")  # <-- update filename if needed

# Create side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Latency vs Size ---
for mix, group in df.groupby("rw_mix"):
    axes[0].plot(group["size"], group["latency_ns"], marker="o", label=mix)

axes[0].set_xscale("log")
axes[0].set_xlabel("Size (KiB)")
axes[0].set_ylabel("Latency (ns)")
axes[0].set_title("Latency vs Size")
axes[0].grid(True)
axes[0].legend()

# --- Bandwidth vs Size ---
for mix, group in df.groupby("rw_mix"):
    axes[1].plot(group["size"], group["bandwidth_MBps"], marker="o", label=mix)

axes[1].set_xscale("log")
axes[1].set_xlabel("Size (KiB)")
axes[1].set_ylabel("Bandwidth (MB/s)")
axes[1].set_title("Bandwidth vs Size")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/latency_bandwidth_rwmix_plots.png", dpi=300)
plt.show()

print("Plots saved as latency_bandwidth_plots.png")
