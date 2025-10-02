import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("data/mlc_patterns.csv")

# Ensure plots directory exists
import os
os.makedirs("plots", exist_ok=True)

# Plot bandwidth vs stride for each R/W mix
for pat in df["pattern"].unique():
    sub = df[df.pattern == pat]
    plt.figure()
    for mix in sub["mix"].unique():
        sub2 = sub[sub.mix == mix]
        plt.plot(
            sub2["stride_B"],
            sub2["MBps"],
            marker="o",
            label=mix
        )
    plt.xscale("log")
    plt.xlabel("Stride (Bytes)")
    plt.ylabel("Bandwidth (MB/s)")
    plt.title(f"Pattern = {pat}")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig(f"plots/MBps_{pat}.png", dpi=160)
    plt.close()
