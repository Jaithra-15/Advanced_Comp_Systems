import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_and_aggregate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={
        "gflops": "GFLOPS",
        "gibps": "GIBPS",
        "cpe": "CPE"
    })

    # Aggregate mean across trials
    agg = (df.groupby(["mode","access","stride","n"], as_index=False)
             .agg(GFLOPS=("GFLOPS","mean"),
                  GIBPS=("GIBPS","mean"),
                  CPE=("CPE","mean")))
    return agg

def style(mode: str):
    return ("-", 2.6) if mode.lower()=="simd" else ("--", 1.7)

def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, outfile: Path, N_sel: int):
    plt.figure(figsize=(8.5,6))
    colors = {"unit": "tab:blue", "strided": "tab:orange", "gather": "tab:green"}

    for access in ["unit","strided","gather"]:
        for mode in ["scalar","simd"]:
            sub = df[(df["access"]==access) & (df["mode"].str.lower()==mode) & (df["n"]==N_sel)]
            if sub.empty:
                continue
            x = sub["stride"].to_numpy()
            y = sub[metric].to_numpy()
            order = np.argsort(x)
            x, y = x[order], y[order]
            ls, lw = style(mode)
            plt.plot(x, y, ls, linewidth=lw, color=colors[access],
                     label=f"{access}, {mode.upper()}")

    plt.xscale("log", base=2)
    plt.xlabel("Stride (elements) [log2]")
    plt.ylabel(ylabel)
    plt.title(f"{metric} vs Stride (N={N_sel:,})")
    plt.legend(ncols=2, fontsize=9)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    print(f"[saved] {outfile}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="stride_gather_sweep.csv", help="Input CSV")
    ap.add_argument("--kernel", default=None, help="Optional kernel filter")
    ap.add_argument("--N", type=int, default=None, help="Select N (default=max N)")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_and_aggregate(args.csv)

    if args.kernel and "kernel" in df.columns:
        df = df[df["kernel"]==args.kernel]

    N_sel = args.N if args.N else df["n"].max()
    # snap to nearest available N
    N_sel = df.iloc[(df["n"]-N_sel).abs().argsort()].iloc[0]["n"]

    plot_metric(df, "GIBPS", "GiB/s (effective bandwidth)", outdir/"stride_gather_bandwidth.png", N_sel)
    plot_metric(df, "CPE", "Cycles per Element (CPE)", outdir/"stride_gather_cpe.png", N_sel)
    plot_metric(df, "GFLOPS", "GFLOP/s", outdir/"stride_gather_gflops.png", N_sel)

if __name__ == "__main__":
    main()
