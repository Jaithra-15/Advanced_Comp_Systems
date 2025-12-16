import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_results.py results.csv FREQ_HZ")
        sys.exit(1)

    results_path = sys.argv[1]
    freq_hz = float(sys.argv[2])

    df = pd.read_csv(results_path)

    # Average repetitions by (impl,mode,keys,threads)
    g = df.groupby(["impl", "mode", "keys", "threads"], as_index=False).mean(numeric_only=True)

    # Derived cycles from fixed frequency
    g["est_cycles"] = g["seconds"] * freq_hz
    g["cycles_per_op"] = g["est_cycles"] / g["ops"]

    # Choose largest key size for main plots
    keys_for_plots = int(g["keys"].max())

    def plot_lines(sub, x, y, title, ylabel, outpng):
        for impl in ["coarse", "fine"]:
            s = sub[sub["impl"] == impl].sort_values(x)
            plt.plot(s[x], s[y], marker="o", label=impl)
        plt.xlabel(x.capitalize())
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outpng)
        plt.clf()

    # Throughput per workload
    for mode in ["lookup", "insert", "mixed"]:
        sub = g[(g["mode"] == mode) & (g["keys"] == keys_for_plots)].copy()
        plot_lines(
            sub, "threads", "throughput",
            f"Throughput vs Threads ({mode}, keys={keys_for_plots})",
            "Throughput (ops/s)",
            f"fig_{mode}_throughput.png"
        )

    # Speedup vs 1 thread, per impl+mode
    sub = g[g["keys"] == keys_for_plots].copy()

    def add_speedup(grp):
        base = grp.loc[grp["threads"] == 1, "throughput"].iloc[0]
        grp = grp.copy()
        grp["speedup"] = grp["throughput"] / base
        return grp

    sub = sub.groupby(["impl", "mode"], group_keys=False).apply(add_speedup)

    for mode in ["lookup", "insert", "mixed"]:
        for impl in ["coarse", "fine"]:
            s = sub[(sub["mode"] == mode) & (sub["impl"] == impl)].sort_values("threads")
            plt.plot(s["threads"], s["speedup"], marker="o", label=f"{impl}-{mode}")

    plt.xlabel("Threads")
    plt.ylabel("Speedup vs 1 thread")
    plt.title(f"Speedup vs Threads (keys={keys_for_plots})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("fig_speedup.png")
    plt.clf()

    # Cycles/op per workload (derived)
    for mode in ["lookup", "insert", "mixed"]:
        sub = g[(g["mode"] == mode) & (g["keys"] == keys_for_plots)].copy()
        plot_lines(
            sub, "threads", "cycles_per_op",
            f"Estimated Cycles/op vs Threads ({mode}, keys={keys_for_plots})",
            "Estimated cycles per op",
            f"fig_{mode}_cycles_per_op.png"
        )

    # Optional: show how results change with key size at fixed thread count
    # (useful to discuss cache / memory effects)
    fixed_threads = 4 if 4 in set(g["threads"]) else int(sorted(set(g["threads"]))[0])
    for mode in ["lookup", "insert", "mixed"]:
        sub = g[(g["mode"] == mode) & (g["threads"] == fixed_threads)].copy()
        for impl in ["coarse", "fine"]:
            s = sub[sub["impl"] == impl].sort_values("keys")
            plt.plot(s["keys"], s["throughput"], marker="o", label=impl)
        plt.xscale("log")
        plt.xlabel("Keys (log scale)")
        plt.ylabel("Throughput (ops/s)")
        plt.title(f"Throughput vs Keyset Size ({mode}, threads={fixed_threads})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig_{mode}_throughput_vs_keys_t{fixed_threads}.png")
        plt.clf()

    print("Wrote plots:")
    print("  fig_<mode>_throughput.png (lookup/insert/mixed)")
    print("  fig_speedup.png")
    print("  fig_<mode>_cycles_per_op.png (derived from seconds * freq)")
    print("  fig_<mode>_throughput_vs_keys_tX.png")

if __name__ == "__main__":
    main()

