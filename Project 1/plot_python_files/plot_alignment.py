import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_aggregate(path: str, simd_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure correct column casing
    df = df.rename(columns={c.lower(): c for c in df.columns})
    df.rename(columns={"gflops": "GFLOPs", "cpe": "CPE"}, inplace=True)
    df["simd"] = simd_label
    # Group by N, misaligned, tail_multiple
    agg = df.groupby(["N", "misaligned", "tail_multiple"]).agg(
        GFLOPs=("GFLOPs", "mean"),
        CPE=("CPE", "mean")
    ).reset_index()
    agg["simd"] = simd_label
    return agg

def label_variant(simd: str, mis: int, tail: int) -> str:
    return f"{simd}, {'misaligned' if mis else 'aligned'}, {'tail-multiple' if tail else 'tail-masked'}"

def plot_metric(df_all: pd.DataFrame, metric: str, ylabel: str, outfile: str):
    plt.figure(figsize=(8,6))
    colors = ["tab:blue","tab:orange","tab:green","tab:red"]
    variants = [(0,1),(1,1),(0,0),(1,0)]  # aligned/tail-multiple, misaligned/tail-multiple, aligned/tail-masked, misaligned/tail-masked

    for idx, (mis, tail) in enumerate(variants):
        for simd in ["SIMD","No-SIMD"]:
            sub = df_all[(df_all["misaligned"]==mis) &
                         (df_all["tail_multiple"]==tail) &
                         (df_all["simd"]==simd)]
            if sub.empty:
                continue
            x = sub["N"].to_numpy()
            y = sub[metric].to_numpy()
            linestyle = "-" if simd=="SIMD" else "--"
            linewidth = 2.5 if simd=="SIMD" else 1.5
            plt.plot(x, y, linestyle=linestyle, linewidth=linewidth,
                     color=colors[idx],
                     label=label_variant(simd, mis, tail))

    plt.xscale("log")
    plt.xlabel("N (elements)")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs N")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    print(f"[saved] {outfile}")

def main():
    simd = load_and_aggregate("saxpy_simd.csv", "SIMD")
    nosimd = load_and_aggregate("saxpy_scalar.csv", "No-SIMD")
    df_all = pd.concat([simd, nosimd], ignore_index=True)

    plot_metric(df_all, "GFLOPs", "GFLOP/s", "plot_saxpy_gflops.png")
    plot_metric(df_all, "CPE", "Cycles per Element (CPE)", "plot_saxpy_cpe.png")

if __name__ == "__main__":
    main()
