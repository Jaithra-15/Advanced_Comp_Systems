import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
df_std = pd.read_csv("kernel_sweep_standard.csv")
df_new = pd.read_csv("kernel_sweep.csv")

# Pick kernel
kernel_name = df_std["kernel"].iloc[0]
df_std = df_std[df_std["kernel"] == kernel_name]
df_new = df_new[df_new["kernel"] == kernel_name]

# Helper: drop top 2 and bottom 2 before computing mean/std
def trimmed_stats(group):
    def trim(col):
        vals = group[col].sort_values()
        if len(vals) > 4:
            vals = vals.iloc[2:-2]
        return vals
    return pd.Series({
        "GFLOPs_mean": trim("GFLOPs").mean(),
        "GFLOPs_std": trim("GFLOPs").std(),
        "CPE_mean": trim("CPE").mean(),
        "CPE_std": trim("CPE").std(),
    })

# Compute stats
std_stats = df_std.groupby("N").apply(trimmed_stats)
new_stats = df_new.groupby("N").apply(trimmed_stats)

# Transition points
transitions = {"L1": 40960, "L2": 109226, "LLC": 2097152, "DRAM": 11184810.7}

# Plot function
def plot_with_errorbars(x, mean, std, label, color, marker):
    plt.plot(x, mean, marker=marker, color=color, label=label)
    for xi, m, s in zip(x, mean, std):
        plt.plot([xi, xi], [m - s, m + s], color=color, linewidth=1)
        dash_width = xi * 0.05
        plt.plot([xi - dash_width, xi + dash_width], [m - s, m - s], color=color, linewidth=1)
        plt.plot([xi - dash_width, xi + dash_width], [m + s, m + s], color=color, linewidth=1)

def add_cache_annotations():
    ymin, ymax = plt.ylim()
    for name, x in transitions.items():
        plt.axvline(x=x, linestyle=":", color="gray", linewidth=1)
        plt.text(x, ymax*0.95, name, rotation=90, va="top", ha="right", color="gray", fontsize=9)

# ---- GFLOPs plot ----
plt.figure()
plot_with_errorbars(std_stats.index, std_stats["GFLOPs_mean"], std_stats["GFLOPs_std"],
                    "no simd", "tab:blue", "o")
plot_with_errorbars(new_stats.index, new_stats["GFLOPs_mean"], new_stats["GFLOPs_std"],
                    "simd", "tab:orange", "x")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("GFLOP/s")
plt.title(f"GFLOP/s vs N (kernel={kernel_name})")
add_cache_annotations()
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot_gflops.png", dpi=150)

# ---- CPE plot ----
plt.figure()
plot_with_errorbars(std_stats.index, std_stats["CPE_mean"], std_stats["CPE_std"],
                    "no simd", "tab:blue", "o")
plot_with_errorbars(new_stats.index, new_stats["CPE_mean"], new_stats["CPE_std"],
                    "simd", "tab:orange", "x")
plt.xscale("log")
plt.xlabel("N")
plt.ylabel("Cycles per Element (CPE)")
plt.title(f"CPE vs N (kernel={kernel_name})")
add_cache_annotations()
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("plot_cpe.png", dpi=150)

print("Plots saved!")
