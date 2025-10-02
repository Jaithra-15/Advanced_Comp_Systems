#!/usr/bin/env python3
"""
make_full_sweep_plots.py

Reads two CSVs:
  - full_sweep_simd.csv
  - full_sweep_scalar.csv
with schema:
  kernel,dtype,N,trial,misaligned,tail_multiple,GFLOPs,CPE

Does the following:
  1) Removes top 5% and bottom 5% outliers by GFLOPs within each
     (mode,kernel,dtype,N,misaligned,tail_multiple) group and recomputes means.
  2) Generates plots with x-axis = element size (N):
     a) Alignment & Tail Handling: GFLOPs on y-axis. For EACH kernel, one figure
        showcasing the FOUR scenarios of (misaligned × tail_multiple):
          {(0,0), (1,0), (0,1), (1,1)}
        We draw SIMD (solid) and Scalar (dashed) lines for each scenario.
     b) Data Type Comparison: For EACH dtype, one figure plotting GFLOPs vs N
        with lines for each kernel. SIMD (solid), Scalar (dashed).
        For clarity, this plot averages across the 4 alignment/tail scenarios
        at each (mode, kernel, dtype, N).
     c) Speed Ratio: For EACH (dtype, misaligned, tail_multiple), one figure with
        y = Scalar/SIMD GFLOPs ratio vs N, lines per kernel. A horizontal y=1 line
        is drawn as reference.

Usage:
  python make_full_sweep_plots.py \
      --simd full_sweep_simd.csv \
      --scalar full_sweep_scalar.csv \
      --outdir plots

Notes:
  - Requires: pandas, numpy, matplotlib
  - No seaborn; no explicit colors; one chart per figure.
"""

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser(description="Generate SIMD vs Scalar sweep plots with outlier trimming.")
    ap.add_argument("--simd", type=str, default="full_sweep_simd.csv", help="Path to SIMD CSV")
    ap.add_argument("--scalar", type=str, default="full_sweep_scalar.csv", help="Path to Scalar CSV")
    ap.add_argument("--outdir", type=str, default="plots", help="Directory to save plots and processed CSVs")
    ap.add_argument("--trim_q", type=float, default=0.05, help="Quantile for bottom/top trimming (default 0.05 => 5%%)")
    return ap.parse_args()

def ensure_numeric(df):
    # Coerce types
    int_cols = ["N","trial","misaligned","tail_multiple"]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(int)
    for c in ["GFLOPs","CPE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def trim_and_mean(group, q=0.05):
    # If too few samples to trim, just average all
    if len(group) < 5:
        return pd.Series({
            "mean_GFLOPs": group["GFLOPs"].mean(),
            "mean_CPE": group["CPE"].mean(),
            "count": len(group)
        })
    lower = group["GFLOPs"].quantile(q)
    upper = group["GFLOPs"].quantile(1.0 - q)
    trimmed = group[(group["GFLOPs"] >= lower) & (group["GFLOPs"] <= upper)]
    return pd.Series({
        "mean_GFLOPs": trimmed["GFLOPs"].mean(),
        "mean_CPE": trimmed["CPE"].mean(),
        "count": len(trimmed)
    })

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    simd = pd.read_csv(args.simd)
    scalar = pd.read_csv(args.scalar)
    simd["mode"] = "SIMD"
    scalar["mode"] = "Scalar"

    ensure_numeric(simd)
    ensure_numeric(scalar)

    df = pd.concat([simd, scalar], ignore_index=True)

    # Outlier trimming & mean aggregation
    group_cols = ["mode","kernel","dtype","N","misaligned","tail_multiple"]
    agg = df.groupby(group_cols, dropna=False).apply(trim_and_mean, q=args.trim_q).reset_index()

    # Save processed means for transparency
    agg.to_csv(outdir / "trimmed_means.csv", index=False)

    # ---- 1) Alignment & Tail Handling: per kernel, 4 scenarios × {SIMD,Scalar} lines ----
    scenarios = [(0,0), (1,0), (0,1), (1,1)]
    scenario_labels = {
        (0,0): "aligned, tail-multiple",
        (1,0): "misaligned, tail-multiple",
        (0,1): "aligned, tail-not-multiple",
        (1,1): "misaligned, tail-not-multiple",
    }

    kernels = sorted(agg["kernel"].dropna().unique())
    dtypes = sorted(agg["dtype"].dropna().unique())

    # We will average across dtypes for this visualization? The user didn't specify;
    # but typically we want to keep dtype effects separate. The request says: "with all three kernels"
    # not "per dtype" here. To keep the plot readable, average across dtypes for each scenario.
    for kernel in kernels:
        fig = plt.figure()
        ax = plt.gca()
        for mis, tail in scenarios:
            for mode, linestyle in [("SIMD","-"), ("Scalar","--")]:
                sub = agg[(agg["kernel"]==kernel) &
                          (agg["misaligned"]==mis) &
                          (agg["tail_multiple"]==tail) &
                          (agg["mode"]==mode)]
                if sub.empty:
                    continue
                # Average across dtype at each N to avoid 3x duplication per scenario
                sub_avg = sub.groupby("N", as_index=False)["mean_GFLOPs"].mean().sort_values("N")
                label = f"{scenario_labels[(mis,tail)]} ({mode})"
                ax.plot(sub_avg["N"].values, sub_avg["mean_GFLOPs"].values, linestyle=linestyle, label=label)

        ax.set_xscale("log")
        ax.set_xlabel("Elements (N)")
        ax.set_ylabel("GFLOPs")
        ax.set_title(f"Alignment & Tail Handling — {kernel}")
        ax.legend(fontsize=8, ncol=2)
        fig.savefig(outdir / f"alignment_tail_{kernel}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    # ---- 2) Data type comparison: per dtype; lines per kernel; SIMD solid, Scalar dashed ----
    for dtype in dtypes:
        fig = plt.figure()
        ax = plt.gca()
        for kernel in kernels:
            for mode, linestyle in [("SIMD","-"), ("Scalar","--")]:
                sub = agg[(agg["dtype"]==dtype) & (agg["kernel"]==kernel) & (agg["mode"]==mode)]
                if sub.empty:
                    continue
                # Average across 4 alignment/tail scenarios for a clean comparison
                sub_avg = sub.groupby("N", as_index=False)["mean_GFLOPs"].mean().sort_values("N")
                label = f"{kernel} ({mode})"
                ax.plot(sub_avg["N"].values, sub_avg["mean_GFLOPs"].values, linestyle=linestyle, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Elements (N)")
        ax.set_ylabel("GFLOPs")
        ax.set_title(f"Data Type Comparison — {dtype}")
        ax.legend(fontsize=9, ncol=2)
        fig.savefig(outdir / f"dtype_comp_{dtype}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    # ---- 3) Speed ratio: Scalar/SIMD vs N, per (dtype, misaligned, tail), lines per kernel ----
    pivot_cols = ["kernel","dtype","N","misaligned","tail_multiple"]
    pivoted = agg.pivot_table(index=pivot_cols, columns="mode", values="mean_GFLOPs")
    pivoted = pivoted.dropna(subset=["SIMD","Scalar"]).reset_index()
    pivoted["ratio_scalar_over_simd"] = pivoted["Scalar"] / pivoted["SIMD"]

    for dtype in dtypes:
        for mis in [0,1]:
            for tail in [0,1]:
                fig = plt.figure()
                ax = plt.gca()
                sub = pivoted[(pivoted["dtype"]==dtype) &
                              (pivoted["misaligned"]==mis) &
                              (pivoted["tail_multiple"]==tail)]
                if sub.empty:
                    plt.close(fig)
                    continue
                for kernel in kernels:
                    s = sub[sub["kernel"]==kernel].sort_values("N")
                    if s.empty:
                        continue
                    ax.plot(s["N"].values, s["ratio_scalar_over_simd"].values, label=kernel)
                ax.set_xscale("log")
                ax.set_xlabel("Elements (N)")
                ax.set_ylabel("Speed (Scalar/SIMD)")
                tail_label = "tail-multiple" if tail==1 else "tail-not-multiple"
                ax.set_title(f"Speed Ratio — dtype {dtype}, misaligned={mis}, {tail_label}")
                ax.axhline(1.0, linestyle=":")
                ax.legend(fontsize=9)
                fig.savefig(outdir / f"ratio_{dtype}_mis{mis}_tail{tail}.png", bbox_inches="tight", dpi=160)
                plt.close(fig)

    print(f"Done. Plots and processed CSV saved to: {outdir}")

if __name__ == "__main__":
    main()
