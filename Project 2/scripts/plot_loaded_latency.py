import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/mlc_loaded_latency.csv")
plt.figure()
plt.plot(df["throughput_MBps"], df["avg_latency_ns"], marker="o")
plt.xlabel("Throughput (MB/s)")
plt.ylabel("Avg Latency (ns)")
plt.title("Throughput–Latency (Loaded Latency)")
plt.grid(True)
# Mark knee: pick argmax of curvature (simple heuristic)
import numpy as np
x=df["throughput_MBps"].values; y=df["avg_latency_ns"].values
knee_idx = np.argmax(np.gradient(np.gradient(y)))
plt.scatter([x[knee_idx]],[y[knee_idx]])
plt.annotate("knee", (x[knee_idx], y[knee_idx]))
plt.savefig("plots/loaded_latency_knee.png", dpi=160)
