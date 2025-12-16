#!/usr/bin/env python3
import argparse
import os
import csv
import glob
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def i(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def pick_csv(user_csv: str) -> str:
    if user_csv and user_csv.strip():
        return user_csv.strip()

    for c in ["results/results_a2.csv", "results.csv"]:
        if os.path.exists(c):
            return c

    gl = glob.glob("results/*.csv")
    if gl:
        gl.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return gl[0]

    raise FileNotFoundError("No CSV found. Pass --csv or run ./run_a2.sh first.")


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as fobj:
        rdr = csv.DictReader(fobj)
        for r in rdr:
            rows.append(r)
    return rows


def median_iqr(vs):
    arr = np.array(vs, dtype=float)
    med = float(np.median(arr))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return med, p25, p75


def savefig(outdir, name):
    p = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[plot] wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to results CSV (default: auto-detect)")
    ap.add_argument("--outdir", default="results", help="Output directory")
    ap.add_argument("--threads", type=int, default=8, help="Thread count to plot (default: 8)")
    ap.add_argument("--gemm_variant", default="simd", choices=["scalar", "simd"], help="GEMM variant")
    ap.add_argument("--spmm_variant", default="simd", choices=["scalar", "simd"], help="SpMM variant")
    ap.add_argument("--spmm_density", type=float, default=0.01, help="SpMM density for working-set sweep")
    ap.add_argument("--spmm_n", type=int, default=256, help="SpMM n for working-set sweep (default: 256)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csv_path = pick_csv(args.csv)
    print(f"[plot] using CSV: {csv_path}")
    rows = read_rows(csv_path)
    if not rows:
        print("[plot] CSV empty.")
        return

    # STREAM reference bandwidth (if present)
    stream_bw = 0.0
    for r in rows:
        if r.get("kernel", "") == "stream":
            stream_bw = max(stream_bw, f(r.get("bandwidth_GBps", "0")))
    # If stream isn't present, that's okay; we'll skip the reference line
    has_stream = stream_bw > 0.0

    # Working-set definition (matches your run_a2.sh design):
    # GEMM: m=k=n=size, kernel=gemm, variant=simd, pattern=uniform
    # SpMM: m=k=size, n=256, density=0.01, kernel=spmm_csr, variant=simd, layoutB=row, pattern=uniform

    def keep_gemm(r):
        if r.get("kernel") != "gemm":
            return False
        if r.get("variant") != args.gemm_variant:
            return False
        if r.get("pattern", "uniform") != "uniform":
            return False
        if r.get("layoutB", "row") != "row":
            return False
        if i(r.get("threads")) != args.threads:
            return False
        m = i(r.get("m"))
        return (m == i(r.get("k")) and m == i(r.get("n")))

    def keep_spmm(r):
        if r.get("kernel") != "spmm_csr":
            return False
        if r.get("variant") != args.spmm_variant:
            return False
        if r.get("pattern", "uniform") != "uniform":
            return False
        if r.get("layoutB", "row") != "row":
            return False
        if i(r.get("threads")) != args.threads:
            return False
        if i(r.get("n")) != args.spmm_n:
            return False
        if abs(f(r.get("density")) - args.spmm_density) > 1e-12:
            return False
        m = i(r.get("m"))
        return (m == i(r.get("k")))

    gemm_rows = [r for r in rows if keep_gemm(r)]
    spmm_rows = [r for r in rows if keep_spmm(r)]

    def agg_by_size(rs, value_key):
        mp = defaultdict(list)
        for r in rs:
            mp[i(r.get("m"))].append(f(r.get(value_key)))
        out = []
        for sz, vs in mp.items():
            med, p25, p75 = median_iqr(vs)
            out.append((sz, med, p25, p75))
        out.sort(key=lambda x: x[0])
        return out

    g_gflops = agg_by_size(gemm_rows, "gflops")
    s_gflops = agg_by_size(spmm_rows, "gflops")

    g_bw = agg_by_size(gemm_rows, "bandwidth_GBps")
    s_bw = agg_by_size(spmm_rows, "bandwidth_GBps")

    # ---- Plot 1: GFLOP/s vs size ----
    if g_gflops or s_gflops:
        plt.figure()
        if g_gflops:
            xs = [x[0] for x in g_gflops]
            ys = [x[1] for x in g_gflops]
            plt.plot(xs, ys, marker="o", label=f"GEMM ({args.gemm_variant}, {args.threads}t)")
        if s_gflops:
            xs = [x[0] for x in s_gflops]
            ys = [x[1] for x in s_gflops]
            plt.plot(xs, ys, marker="o",
                     label=f"CSR-SpMM ({args.spmm_variant}, n={args.spmm_n}, d={args.spmm_density:g}, {args.threads}t)")
        plt.xlabel("Size parameter (GEMM: m=k=n; SpMM: m=k)")
        plt.ylabel("GFLOP/s (median over runs)")
        plt.title("Working-set transitions: performance vs size")
        plt.grid(True)
        plt.legend()
        savefig(args.outdir, "fig_workingset_gflops.png")
    else:
        print("[plot] warning: no rows found for working-set GFLOP/s plot (did you run the size sweep?).")

    # ---- Plot 2: Bandwidth vs size ----
    if g_bw or s_bw:
        plt.figure()
        if g_bw:
            xs = [x[0] for x in g_bw]
            ys = [x[1] for x in g_bw]
            plt.plot(xs, ys, marker="o", label=f"GEMM BW est ({args.gemm_variant}, {args.threads}t)")
        if s_bw:
            xs = [x[0] for x in s_bw]
            ys = [x[1] for x in s_bw]
            plt.plot(xs, ys, marker="o",
                     label=f"CSR-SpMM BW est ({args.spmm_variant}, n={args.spmm_n}, d={args.spmm_density:g}, {args.threads}t)")
        if has_stream:
            plt.axhline(stream_bw, linestyle="--", label=f"STREAM triad ~ {stream_bw:.1f} GB/s")
        plt.xlabel("Size parameter (GEMM: m=k=n; SpMM: m=k)")
        plt.ylabel("Bandwidth (GB/s)")
        plt.title("Working-set transitions: bandwidth vs size")
        plt.grid(True)
        plt.legend()
        savefig(args.outdir, "fig_workingset_bandwidth.png")
    else:
        print("[plot] warning: no rows found for working-set bandwidth plot (bandwidth_GBps missing/zero?).")


if __name__ == "__main__":
    main()

