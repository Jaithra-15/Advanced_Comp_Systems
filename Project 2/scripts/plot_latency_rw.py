import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# Load the latency results
df = pd.read_csv("data/mlc_latency_rw.csv")

plt.figure()

# Plot one line per rw_mix
for mix, group in df.groupby("rw_mix"):
    plt.plot(group["size_KiB"]/1024, group["latency_ns"], marker="o", label=mix)

plt.xscale("log")
plt.xlabel("Working set size (MiB)")
plt.ylabel("Latency (ns)")
plt.title("Idle Latency vs Working Set Size (R/W mixes)")
plt.grid(True, which="both")
plt.legend()

# Annotate approximate cache levels
for label, mb in [("L1", 0.032), ("L2", 0.256), ("L3", 4), ("DRAM", 512)]:
    plt.axvline(mb, linestyle="--", color="red", alpha=0.6)
    ymax = df["latency_ns"].max()
    plt.text(mb*1.05, ymax*0.8, label, rotation=90)

plt.savefig("plots/latency_vs_workingset_rw.png", dpi=160)
plt.close()
