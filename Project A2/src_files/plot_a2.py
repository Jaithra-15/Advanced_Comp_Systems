#!/usr/bin/env python3
# plot_a2.py — Project A2 plotting (robust + WSL/headless friendly + sane thread scaling plots)

import argparse
import os
import glob
import csv
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend (prevents Qt/Wayland errors on WSL/headless shells)
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


def read_rows(path):
    rows = []
    with open(path, "r", newline="") as fobj:
        rdr = csv.DictReader(fobj)
        for r in rdr:
            rows.append(r)
    return rows


def pick_csv(user_csv: str) -> str:
    if user_csv is not None and user_csv.strip() != "":
        return user_csv.strip()

    for c in ["results/results_a2.csv", "results.csv"]:
        if os.path.exists(c):
            return c

    gl = glob.glob("results/*.csv")
    if gl:
        gl.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return gl[0]

    raise FileNotFoundError("No CSV found. Expected results/results_a2.csv (run ./run_a2.sh first).")


def savefig(outdir, name):
    p = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[plot] wrote {p}")


def agg_by_threads(rows, value_key):
    """
    Returns dict: threads -> (median, p25, p75, count)
    """
    mp = defaultdict(list)
    for r in rows:
        mp[i(r.get("threads", 0))].append(f(r.get(value_key, 0.0)))
    out = {}
    for t, vs in mp.items():
        arr = np.array(vs, dtype=float)
        med = float(np.median(arr))
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        out[t] = (med, p25, p75, len(arr))
    return out


def choose_best_threadcase(rows, kernel, fixed_filters, config_keys, require_variants=("scalar", "simd")):
    """
    Pick the (config tuple) that maximizes number of unique thread points, preferring configs
    where BOTH required variants exist with multiple thread points.
    """
    # apply fixed filters
    cand = []
    for r in rows:
        if r.get("kernel") != kernel:
            continue
        ok = True
        for k, v in fixed_filters.items():
            if v is None:
                continue
            if k in ("m", "k", "n"):
                if i(r.get(k)) != int(v):
                    ok = False
                    break
            elif k == "density":
                if abs(f(r.get("density")) - float(v)) > 1e-12:
                    ok = False
                    break
            else:
                if r.get(k) != v:
                    ok = False
                    break
        if ok:
            cand.append(r)

    if not cand:
        return None, []

    # group by config keys
    groups = defaultdict(list)
    for r in cand:
        cfg = tuple(
            i(r.get(k)) if k in ("m", "k", "n", "tileM", "tileN", "tileK", "jblock") else
            f(r.get(k)) if k == "density" else
            r.get(k)
            for k in config_keys
        )
        groups[cfg].append(r)

    best_cfg = None
    best_score = (-1, -1, -1)  # (min_threads_across_variants, total_threads_union, total_rows)
    best_rows = None

    for cfg, rs in groups.items():
        by_var = defaultdict(list)
        for r in rs:
            by_var[r.get("variant", "")].append(r)

        # require variants present
        if any(v not in by_var for v in require_variants):
            continue

        thread_sets = []
        total_rows = 0
        for v in require_variants:
            ts = set(i(x.get("threads")) for x in by_var[v])
            thread_sets.append(ts)
            total_rows += len(by_var[v])

        min_threads = min(len(ts) for ts in thread_sets)
        union_threads = len(set().union(*thread_sets))

        score = (min_threads, union_threads, total_rows)
        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_rows = rs

    if best_cfg is None:
        # fallback: just pick config with most thread points for *any* variant
        best_cfg = max(groups.keys(), key=lambda c: len(set(i(r.get("threads")) for r in groups[c])))
        best_rows = groups[best_cfg]

    return best_cfg, best_rows


def plot_thread_scaling(rows, outdir, title, fig_name):
    """
    rows must include 'variant' and 'threads'.
    """
    by_var = defaultdict(list)
    for r in rows:
        by_var[r.get("variant", "")].append(r)

    plt.figure()

    for var in ["scalar", "simd"]:
        rs = by_var.get(var, [])
        if not rs:
            continue
        mp = agg_by_threads(rs, "gflops")
        ts = sorted(mp.keys())
        ys = [mp[t][0] for t in ts]
        yerr_low = [mp[t][0] - mp[t][1] for t in ts]
        yerr_high = [mp[t][2] - mp[t][0] for t in ts]

        # Only draw a line if there are 2+ points; otherwise marker only
        if len(ts) >= 2:
            plt.errorbar(ts, ys, yerr=[yerr_low, yerr_high], marker="o", linestyle="-", capsize=3, label=var)
        else:
            plt.errorbar(ts, ys, yerr=[yerr_low, yerr_high], marker="o", linestyle="None", capsize=3, label=var)

    plt.xlabel("Threads")
    plt.ylabel("GFLOP/s (median, p25–p75)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    savefig(outdir, fig_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to results CSV (default: auto-detect)")
    ap.add_argument("--outdir", default="results", help="Output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    csv_path = pick_csv(args.csv)
    print(f"[plot] using CSV: {csv_path}")
    rows = read_rows(csv_path)
    if not rows:
        print("[plot] CSV has no rows; nothing to plot.")
        return

    # STREAM bandwidth reference
    stream_bw = 0.0
    for r in rows:
        if r.get("kernel", "") == "stream":
            stream_bw = max(stream_bw, f(r.get("bandwidth_GBps", "0")))
    if stream_bw <= 0:
        stream_bw = 20.0

    # -----------------------------
    # Figure 1: GEMM thread scaling
    # Auto-pick the GEMM config with the most thread points
    # -----------------------------
    gemm_fixed = {"layoutB": "row", "pattern": "uniform", "density": 1.0}
    gemm_cfg_keys = ("m", "k", "n", "tileM", "tileN", "tileK")
    gemm_cfg, gemm_rows = choose_best_threadcase(rows, "gemm", gemm_fixed, gemm_cfg_keys)

    if gemm_cfg is not None:
        # restrict to chosen config
        chosen = []
        for r in gemm_rows:
            cfg = (i(r.get("m")), i(r.get("k")), i(r.get("n")),
                   i(r.get("tileM")), i(r.get("tileN")), i(r.get("tileK")))
            if cfg == tuple(gemm_cfg[:6]):
                chosen.append(r)

        m, k, n, tM, tN, tK = gemm_cfg[:6]
        plot_thread_scaling(
            chosen,
            args.outdir,
            f"GEMM GFLOP/s vs threads (m={m},k={k},n={n}, tiles={tM}/{tK}/{tN})",
            "fig_gemm_gflops_threads.png",
        )
    else:
        print("[plot] warning: no GEMM rows found for thread scaling")

    # -----------------------------
    # Figure 2: CSR-SpMM thread scaling (FIXED)
    # Auto-pick the SpMM config with the most thread points
    # -----------------------------
    spmm_fixed = {"layoutB": "row", "pattern": "uniform", "density": 0.01}
    spmm_cfg_keys = ("m", "k", "n", "density", "jblock")
    spmm_cfg, spmm_rows = choose_best_threadcase(rows, "spmm_csr", spmm_fixed, spmm_cfg_keys)

    if spmm_cfg is not None:
        chosen = []
        for r in spmm_rows:
            cfg = (i(r.get("m")), i(r.get("k")), i(r.get("n")),
                   f(r.get("density")), i(r.get("jblock")))
            if cfg == tuple(spmm_cfg[:5]):
                chosen.append(r)

        m, k, n, d, jb = spmm_cfg[:5]
        plot_thread_scaling(
            chosen,
            args.outdir,
            f"CSR-SpMM GFLOP/s vs threads (m={m},k={k},n={n}, density={d:g}, jblock={jb}, row-major B)",
            "fig_spmm_gflops_threads.png",
        )
    else:
        print("[plot] warning: no SpMM rows found for thread scaling (density=1%)")

    # -----------------------------------
    # Break-even density (kept simple)
    # -----------------------------------
    # If you want this to reflect your chosen m/k/n, rerun the sweep accordingly.
    target_m, target_k, target_n = 2048, 2048, 512
    target_threads = 8

    def closest_thread_subset(rs, target):
        if not rs:
            return []
        diffs = [abs(i(r.get("threads")) - target) for r in rs]
        md = min(diffs)
        return [r for r in rs if abs(i(r.get("threads")) - target) == md]

    gemm_be = [r for r in rows if r.get("kernel") == "gemm" and r.get("variant") == "simd"
               and i(r.get("m")) == target_m and i(r.get("k")) == target_k and i(r.get("n")) == target_n]
    spmm_be = [r for r in rows if r.get("kernel") == "spmm_csr" and r.get("variant") == "simd"
               and i(r.get("m")) == target_m and i(r.get("k")) == target_k and i(r.get("n")) == target_n
               and r.get("layoutB") == "row" and r.get("pattern") == "uniform"]

    gemm_be = closest_thread_subset(gemm_be, target_threads)
    spmm_be = closest_thread_subset(spmm_be, target_threads)

    if gemm_be and spmm_be:
        gemm_rt = float(np.median([f(r.get("seconds")) for r in gemm_be]))

        mp = defaultdict(list)
        for r in spmm_be:
            mp[f(r.get("density"))].append(f(r.get("seconds")))

        ds = sorted(mp.keys())
        sp_rt = [float(np.median(mp[d])) for d in ds]

        plt.figure()
        plt.axhline(gemm_rt, linestyle="--", label="dense GEMM (fixed)")
        plt.plot(ds, sp_rt, marker="o", label="CSR-SpMM (median)")
        plt.xscale("log")
        plt.xlabel("Density (nnz / (m*k))")
        plt.ylabel("Runtime (seconds)")
        plt.title(f"Density break-even (m={target_m},k={target_k},n={target_n}, ~{target_threads} threads)")
        plt.grid(True, which="both")
        plt.legend()
        savefig(args.outdir, "fig_breakeven_density_runtime.png")

    # Roofline (still simple)
    peak_gflops = 500.0  # placeholder
    pts = []
    for r in rows:
        if r.get("kernel") not in ("gemm", "spmm_csr"):
            continue
        if r.get("variant") != "simd":
            continue
        ai = f(r.get("ai"))
        g = f(r.get("gflops"))
        if ai > 0 and g > 0:
            pts.append((ai, g, r.get("kernel")))

    if pts:
        ai_x = np.logspace(-3, 3, 300)
        roof = np.minimum(peak_gflops, stream_bw * ai_x)

        plt.figure()
        plt.loglog(ai_x, roof, label="Roofline: min(Peak, BW*AI)")
        for ker in ("gemm", "spmm_csr"):
            xs = [a for (a, g, k) in pts if k == ker]
            ys = [g for (a, g, k) in pts if k == ker]
            if xs:
                plt.loglog(xs, ys, marker="o", linestyle="None", label=ker)

        plt.xlabel("Arithmetic Intensity (FLOP/byte) [model]")
        plt.ylabel("Achieved GFLOP/s")
        plt.title(f"Roofline (STREAM BW={stream_bw:.1f} GB/s, Peak={peak_gflops:.0f} GFLOP/s placeholder)")
        plt.grid(True, which="both")
        plt.legend()
        savefig(args.outdir, "fig_roofline.png")

    print("[plot] done")


if __name__ == "__main__":
    main()

