import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/mlc_latency_sweep.csv")

plt.figure()
plt.plot(df["size_KiB"]/1024, df["latency_ns"], marker="o")
plt.xscale("log")
plt.xlabel("Working set size (MiB)")
plt.ylabel("Latency (ns)")
plt.title("Idle Latency vs Working Set Size")
plt.grid(True, which="both")

# Annotate approximate cache levels
for label, mb in [("L1", 0.032), ("L2", 0.256), ("L3", 4), ("DRAM", 512)]:
    plt.axvline(mb, linestyle="--", color="red", alpha=0.6)
    plt.text(mb*1.05, max(df.latency_ns)*0.8, label, rotation=90)

plt.savefig("plots/latency_vs_workingset.png", dpi=160)
plt.close()
