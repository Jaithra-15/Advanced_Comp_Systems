#!/usr/bin/env python3
import sys, glob, os, re
import json
import matplotlib.pyplot as plt
from statistics import mean, stdev

results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
figs_dir = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(figs_dir, exist_ok=True)

def extract_latency(job):
    """
    Compute weighted average latency from fio histogram
    (latency_ns, latency_us, latency_ms if available).
    Returns average latency in microseconds (µs).
    """
    buckets = {}

    # Convert everything into µs
    if "latency_ns" in job:
        for k, v in job["latency_ns"].items():
            buckets[float(k) / 1000.0] = v  # ns → µs

    if "latency_us" in job:
        for k, v in job["latency_us"].items():
            buckets[float(k)] = v  # already µs

    if "latency_ms" in job:
        for k, v in job["latency_ms"].items():
            if k.startswith(">="):
                buckets[k] = v
            else:
                buckets[float(k) * 1000.0] = v  # ms → µs

    if not buckets:
        return None

    avg = 0.0
    total_pct = 0.0
    prev_bound = 0.0

    def sort_key(x):
        if isinstance(x[0], str) and x[0].startswith(">="):
            return float(x[0][2:]) * 1000.0
        return float(x[0])

    for bound, pct in sorted(buckets.items(), key=sort_key):
        pct = float(pct)
        if pct <= 0.0:
            continue
        if isinstance(bound, str) and bound.startswith(">="):
            lower_ms = float(bound[2:])
            midpoint = (lower_ms * 1.5) * 1000.0
        else:
            midpoint = (prev_bound + float(bound)) / 2.0

        avg += midpoint * pct
        total_pct += pct
        prev_bound = float(bound) if not isinstance(bound, str) else prev_bound

    return avg / 100.0 if total_pct > 0 else None

def collect(pattern):
    files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    data = {}
    for f in files:
        with open(f) as fh:
            j = json.load(fh)
        for job in j.get("jobs", []):
            bs = re.search(r"bs_.*_(\d+[kM])_", f)
            bs = bs.group(1) if bs else job.get("job options", {}).get("bs", "unknown")

            kind = "read" if "read" in f and "write" not in f else ("write" if "write" in f else "unknown")
            rwtype = ("rand" if "rand" in f else "seq") + "_" + kind

            read = job.get("read", {}); write = job.get("write", {})
            iops = read.get("iops", 0.0) + write.get("iops", 0.0)
            bw_MBps = (read.get("bw", 0) + write.get("bw", 0)) / 1024.0

            lat_us_val = extract_latency(job)
            if lat_us_val is None:
                lat_us_val = 0.0

            data.setdefault((bs, rwtype), []).append((iops, bw_MBps, lat_us_val))

    agg = {}
    for (bs, rwtype), vals in data.items():
        iops_vals = [v[0] for v in vals]
        bw_vals = [v[1] for v in vals]
        lat_vals = [v[2] for v in vals]
        agg[(bs, rwtype)] = (
            mean(iops_vals), (stdev(iops_vals) if len(iops_vals) > 1 else 0.0),
            mean(bw_vals), (stdev(bw_vals) if len(bw_vals) > 1 else 0.0),
            mean(lat_vals), (stdev(lat_vals) if len(lat_vals) > 1 else 0.0)
        )
    return agg

def bs_key(bs):
    if bs.endswith("k"): return int(bs[:-1]) * 1024
    if bs.endswith("M"): return int(bs[:-1]) * 1024 * 1024
    return int(bs)

# --- gather all results together ---
agg_all = {}
for pattern in ["bs_rand_*_*.json", "bs_seq_*_*.json"]:
    agg_all.update(collect(pattern))

# --- now plot overlays ---
labels = {
    "rand_read": "Random Read",
    "rand_write": "Random Write",
    "seq_read": "Sequential Read",
    "seq_write": "Sequential Write"
}
colors = {
    "rand_read": "tab:blue",
    "rand_write": "tab:orange",
    "seq_read": "tab:green",
    "seq_write": "tab:red"
}

# --- IOPS vs Block Size ---
plt.figure(figsize=(7,5))
for rwtype in labels:
    xs=[]; ys=[]; yerr=[]
    for (bs,k),v in sorted(agg_all.items(), key=lambda kv: bs_key(kv[0][0])):
        if k!=rwtype: continue
        xs.append(bs_key(bs)); ys.append(v[0]); yerr.append(v[1])
    if xs:
        plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", label=labels[rwtype], color=colors[rwtype])
plt.xscale("log", base=2)
plt.xlabel("Block size (bytes)")
plt.ylabel("IOPS")
plt.title("IOPS vs Block Size")
plt.grid(True, which="both", axis="both")
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"iops_vs_bs.png")); plt.close()

# --- Throughput vs Block Size ---
plt.figure(figsize=(7,5))
for rwtype in labels:
    xs=[]; ys=[]; yerr=[]
    for (bs,k),v in sorted(agg_all.items(), key=lambda kv: bs_key(kv[0][0])):
        if k!=rwtype: continue
        xs.append(bs_key(bs)); ys.append(v[2]); yerr.append(v[3])
    if xs:
        plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", label=labels[rwtype], color=colors[rwtype])
plt.xscale("log", base=2)
plt.xlabel("Block size (bytes)")
plt.ylabel("Throughput (MB/s)")
plt.title("Throughput vs Block Size")
plt.grid(True, which="both", axis="both")
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"mbps_vs_bs.png")); plt.close()

# --- Latency vs Block Size ---
plt.figure(figsize=(7,5))
for rwtype in labels:
    xs=[]; ys=[]; yerr=[]
    for (bs,k),v in sorted(agg_all.items(), key=lambda kv: bs_key(kv[0][0])):
        if k!=rwtype: continue
        xs.append(bs_key(bs)); ys.append(v[4]); yerr.append(v[5])
    if xs:
        plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-", label=labels[rwtype], color=colors[rwtype])
plt.xscale("log", base=2)
plt.xlabel("Block size (bytes)")
plt.ylabel("Average latency (µs)")
plt.title("Latency vs Block Size")
plt.grid(True, which="both", axis="both")
plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(figs_dir,"latency_vs_bs.png")); plt.close()
