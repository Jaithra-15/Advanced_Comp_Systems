import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os

# -------------------------------
# 1. Load data
# -------------------------------
base = r"C:\Users\jaith\OneDrive\Documents\CLASSES\New folder (2)"
files = glob.glob(os.path.join(base, "simd_*_*_results.csv"))

df_list = []
for f in files:
    # Expect filename like: simd_40960_saxpy_results.csv
    m = re.match(r".*simd_(\d+)_(\w+)_results\.csv", f)
    if not m:
        continue
    N = int(m.group(1))
    kernel = m.group(2)
    df = pd.read_csv(f)
    df["N"] = N
    df["kernel"] = kernel
    # convert seconds → microseconds
    df["time_us"] = df["time_s"] * 1e6
    df_list.append(df)

if not df_list:
    raise RuntimeError("No matching SIMD CSV files found!")

df = pd.concat(df_list, ignore_index=True)

# -------------------------------
# 2. Drop top 5 and bottom 5 trials per (N, kernel)
# -------------------------------
def trim_outliers(group):
    sorted_vals = group.sort_values("time_us")
    if len(sorted_vals) > 10:   # only trim if enough samples
        return sorted_vals.iloc[5:-5]
    return sorted_vals  # keep as-is if fewer than 10 trials

df_trimmed = df.groupby(["N", "kernel"], group_keys=False).apply(trim_outliers)

# -------------------------------
# 3. Aggregate statistics
# -------------------------------
stats = df_trimmed.groupby(["N", "kernel"]).agg(
    avg_time=("time_us", "mean"),
    std_time=("time_us", "std"),
    min_time=("time_us", "min"),
    max_time=("time_us", "max")
).reset_index()

stats["err_low"] = stats["avg_time"] - stats["min_time"]
stats["err_high"] = stats["max_time"] - stats["avg_time"]

# -------------------------------
# 4. Plot grouped bar chart
# -------------------------------
Ns = sorted(stats["N"].unique())
kernels = ["saxpy", "elemmul", "stencil"]
bar_width = 0.25
x = range(len(Ns))

fig, ax = plt.subplots(figsize=(10,6))

for i, k in enumerate(kernels):
    sub = stats[stats["kernel"] == k].set_index("N")
    heights = [sub.loc[n, "avg_time"] for n in Ns]
    stds = [sub.loc[n, "std_time"] for n in Ns]
    err_low = [sub.loc[n, "err_low"] for n in Ns]
    err_high = [sub.loc[n, "err_high"] for n in Ns]

    xpos = [p + i*bar_width for p in x]
    bars = ax.bar(xpos,
                  heights,
                  width=bar_width,
                  yerr=[err_low, err_high],
                  capsize=5,
                  label=k)

    # Add average ± 2*std label above each bar
    for bar, val, std in zip(bars, heights, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{val:.2f} ± {2*std:.2f}",   # µs
                ha="center", va="bottom", fontsize=8)

ax.set_xticks([p + bar_width for p in x])
ax.set_xticklabels(Ns)
ax.set_ylabel("Average runtime (µs)")
ax.set_xlabel("N (size)")
ax.set_title("SIMD Kernel runtimes (outliers removed, min/max error bars)")
ax.legend()
plt.tight_layout()
plt.show()
