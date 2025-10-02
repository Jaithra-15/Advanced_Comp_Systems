#!/usr/bin/env python3
import json, sys, glob, math, statistics, re, os

def load_jobs(paths):
    rows = []
    for p in sorted(paths):
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"[WARN] Failed {p}: {e}", file=sys.stderr)
            continue
        for j in data.get("jobs", []):
            read = j.get("read", {}); write = j.get("write", {})
            # metrics
            iops = (read.get("iops", 0.0) + write.get("iops", 0.0))
            bw_kbps = (read.get("bw", 0) + write.get("bw", 0))
            bw_MBps = bw_kbps/1024.0
            lat = j.get("clat_ns") or j.get("lat_ns") or {}
            lat_us = lat.get("mean", 0.0)/1000.0
            p95 = lat.get("percentile", {}).get("95.000000", float("nan"))/1000.0
            p99 = lat.get("percentile", {}).get("99.000000", float("nan"))/1000.0
            rows.append((p, j["jobname"], iops, bw_MBps, lat_us, p95, p99))
    return rows

def mean_stdev(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals: return float("nan"), float("nan")
    if len(vals) == 1: return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)

def parse_param_from_fname(fname, key):
    # expects names like bs_rand_4k_read_r1_2025.json
    m = re.search(rf"{key}([A-Za-z0-9\.]+)", fname)
    return m.group(1) if m else None

if __name__ == "__main__":
    # utility module; imported by plot scripts
    pass
