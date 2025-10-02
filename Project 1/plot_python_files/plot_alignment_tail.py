
import argparse
import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def trim_outliers(series: pd.Series, k_top=5, k_bottom=5) -> pd.Series:
    vals = series.sort_values().to_list()
    n = len(vals)
    if n == 0:
        return pd.Series([], dtype=float)
    # only trim when we have enough samples
    bt = min(k_bottom, max(0, n // 10)) if n < (k_top + k_bottom) else k_bottom
    tp = min(k_top, max(0, n // 10)) if n < (k_top + k_bottom) else k_top
    start = bt
    end = n - tp
    if end <= start:
        return pd.Series(vals, dtype=float)
    return pd.Series(vals[start:end], dtype=float)

def aggregate_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Group by N, misaligned, tail_multiple. Trim outliers within each group, then compute mean/std.
    rows = []
    for (N, mis, tail), grp in df.groupby(["N", "misaligned", "tail_multiple"]):
        s = trim_outliers(grp[metric])
        if len(s) == 0:
            continue
        rows.append({"N": N, "misaligned": mis, "tail_multiple": tail,
                     "mean": float(s.mean()), "std": float(s.std(ddof=0))})
    out = pd.DataFrame(rows).sort_values("N")
    return out

def label_for_variant(mis: int, tail: int) -> str:
    return f"{'misaligned' if mis else 'aligned'}, {'tail-multiple' if tail else 'tail-masked'}"

def plot_metric(stats: pd.DataFrame, metric_name: str, ylabel: str, outpath: Path):
    plt.figure()
    # Plot 4 variants if present
    variants = [(0,1),(1,1),(0,0),(1,0)]
    markers = ["o", "x", "s", "d"]
    m_idx = 0
    for mis, tail in variants:
        sub = stats[(stats["misaligned"]==mis) & (stats["tail_multiple"]==tail)]
        if sub.empty: 
            continue
        x = sub["N"].to_numpy()
        y = sub["mean"].to_numpy()
        e = sub["std"].to_numpy()
        plt.plot(x, y, marker=markers[m_idx % len(markers)], label=label_for_variant(mis, tail))
        # vertical connector + horizontal dash endpoints for ±std
        for xi, yi, ei in zip(x, y, e):
            lo, hi = yi - ei, yi + ei
            plt.plot([xi, xi], [lo, hi], linewidth=1)
            # dash width as a small fraction of x (works with log x-scale)
            dash_w = xi * 0.04
            plt.plot([xi - dash_w, xi + dash_w], [lo, lo], linewidth=1)
            plt.plot([xi - dash_w, xi + dash_w], [hi, hi], linewidth=1)
        m_idx += 1

    plt.xscale("log")
    plt.xlabel("N (elements)")
    plt.ylabel(ylabel)
    plt.title(f"{metric_name} vs N")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    print(f"[saved] {outpath}")

def geometric_mean(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return float("nan")
    import math
    return math.exp(sum(math.log(v) for v in vals)/len(vals))

def quantify_gaps(stats_gflops: pd.DataFrame) -> str:
    # Compute geo-mean ratios across N for: misalignment penalty, tail penalty, combined
    report_lines = []
    # Helper to get series aligned by N
    def series_for(mis, tail):
        sub = stats_gflops[(stats_gflops["misaligned"]==mis) & (stats_gflops["tail_multiple"]==tail)]
        return sub[["N","mean"]].set_index("N").sort_index()["mean"]

    base = series_for(0,1)  # aligned, tail-multiple (best case)
    if base.empty:
        return "Not enough data to compute gaps."

    def ratio_geo(target):
        joined = pd.concat([base, target], axis=1, join="inner")
        if joined.shape[0] == 0:
            return float("nan")
        b = joined.iloc[:,0].to_numpy()
        t = joined.iloc[:,1].to_numpy()
        # relative throughput = target / base
        ratios = t / b
        return geometric_mean(ratios)

    mis_only = series_for(1,1)
    tail_only = series_for(0,0)
    both     = series_for(1,0)

    gm_mis  = ratio_geo(mis_only)  if not mis_only.empty else float("nan")
    gm_tail = ratio_geo(tail_only) if not tail_only.empty else float("nan")
    gm_both = ratio_geo(both)      if not both.empty     else float("nan")

    def fmt_ratio(r):
        if not (r==r):
            return "n/a"
        return f"{r:.3f}× of aligned baseline  (gap: {(1.0-r)*100:.1f}% slower)"

    report_lines.append("Throughput gap (GFLOP/s), geometric-mean across N:")
    report_lines.append(f"  • Misalignment only  (aligned tail-multiple → misaligned tail-multiple): {fmt_ratio(gm_mis)}")
    report_lines.append(f"  • Tail only          (aligned tail-multiple → aligned tail-masked):     {fmt_ratio(gm_tail)}")
    report_lines.append(f"  • Misalign + Tail    (aligned tail-multiple → misaligned tail-masked):  {fmt_ratio(gm_both)}")
    report_lines.append("Explanations:")
    report_lines.append("  • Prologue/Epilogue (tail): when N % vector_width != 0, compilers peel/clean up or use masked ops → extra control/masking work.")
    report_lines.append("  • Unaligned loads/stores (misaligned): extra µops and cache-line splits vs aligned moves → lower sustained throughput.")
    report_lines.append("  • Combined effects stack; penalties are largest for cache-resident sizes and compress once DRAM-bound.")
    return "\n".join(report_lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="alignment_sweep.csv", help="Input CSV produced by the benchmark")
    ap.add_argument("--kernel", default=None, help="Kernel name to filter (e.g., saxpy)")
    ap.add_argument("--outdir", default=".", help="Directory for plots & report")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    if args.kernel is None and "kernel" in df.columns and not df.empty:
        args.kernel = str(df["kernel"].iloc[0])

    if "kernel" in df.columns and args.kernel is not None:
        df = df[df["kernel"] == args.kernel]

    # Ensure needed columns
    for col in ["N","trial","misaligned","tail_multiple","GFLOPs","CPE"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column in CSV: {col}")

    # types
    for col in ["N","trial","misaligned","tail_multiple"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["GFLOPs","CPE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate with outlier trimming
    stats_g = aggregate_stats(df, "GFLOPs")
    stats_c = aggregate_stats(df, "CPE")

    # Plots
    plot_metric(stats_g, f"GFLOP/s ({args.kernel})", "GFLOP/s", outdir / "plot_alignment_gflops.png")
    plot_metric(stats_c, f"CPE ({args.kernel})", "Cycles per Element (CPE)", outdir / "plot_alignment_cpe.png")

    # Quantify gaps
    report_txt = []
    report_txt.append(f"Kernel: {args.kernel}")
    report_txt.append(quantify_gaps(stats_g))
    report_txt.append("Additional notes:")
    report_txt.append("  • Expect biggest penalties at small/medium N where cache effects dominate.")
    report_txt.append("  • Past LLC/DRAM working-set sizes, all variants compress toward the same bandwidth ceiling.")
    full_report = "\n".join(report_txt)

    rpt = outdir / "alignment_report.txt"
    with open(rpt, "w") as f:
        f.write(full_report)
    print(f"[saved] {rpt}")
    print()
    print(full_report)

if __name__ == "__main__":
    main()
