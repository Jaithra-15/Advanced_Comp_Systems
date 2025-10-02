#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("data/mlc_patterns.csv")

# Make sure size is numeric
df["size"] = df["size"].astype(int)

# Sort for plotting
df = df.sort_values(["size", "pattern"])

# Check coverage: every size should have both seq and rand
sizes = df["size"].unique()
patterns = df["pattern"].unique()
print(f"Sizes tested: {sizes}")
print(f"Patterns: {patterns}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Latency plot ---
for pattern in patterns:
    sub = df[df["pattern"] == pattern]
    axes[0].plot(sub["size"], sub["latency_ns"],
                 marker="o", label=pattern.capitalize())
axes[0].set_xscale("log", base=2)
axes[0].set_xlabel("Block Size (B)")
axes[0].set_ylabel("Latency (ns)")
axes[0].set_title("Loaded Latency vs Size")
axes[0].grid(True, which="both", linestyle="--", alpha=0.6)
axes[0].legend()

# --- Bandwidth plot ---
for pattern in patterns:
    sub = df[df["pattern"] == pattern]
    axes[1].plot(sub["size"], sub["bandwidth_MBps"],
                 marker="s", label=pattern.capitalize())
axes[1].set_xscale("log", base=2)
axes[1].set_xlabel("Block Size (B)")
axes[1].set_ylabel("Bandwidth (MB/s)")
axes[1].set_title("Bandwidth vs Size")
axes[1].grid(True, which="both", linestyle="--", alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig("mlc_seq_rand.png", dpi=150)
plt.show()
