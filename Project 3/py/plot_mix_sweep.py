#!/usr/bin/env python3
import sys, glob, os, re, json
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
        # ns → µs
        for k, v in job["latency_ns"].items():
            buckets[float(k) / 1000.0] = v

    if "latency_us" in job:
        # already µs
        for k, v in job["latency_us"].items():
            buckets[float(k)] = v

    if "latency_ms" in job:
        # ms → µs
        for k, v in job["latency_ms"].items():
            if k.startswith(">="):
                buckets[k] = v  # keep as string, handle later
            else:
                buckets[float(k) * 1000.0] = v

    if not buckets:
        return None

    # Weighted average
    avg = 0.0
    total_pct = 0.0
    prev_bound = 0.0

    # Sort numerically, treating ">=..." last
    def sort_key(x):
        if isinstance(x[0], str) and x[0].startswith(">="):
            return float(x[0][2:]) * 1000.0  # assume ms, convert to µs
        return float(x[0])

    for bound, pct in sorted(buckets.items(), key=sort_key):
        pct = float(pct)
        if pct <= 0.0:
            continue

        if isinstance(bound, str) and bound.startswith(">="):
            # e.g. ">=2000" ms → assume midpoint at 1.5 × lower bound
            lower_ms = float(bound[2:])
            midpoint = (lower_ms * 1.5) * 1000.0  # convert ms → µs
        else:
            midpoint = (prev_bound + float(bound)) / 2.0

        avg += midpoint * pct
        total_pct += pct
        prev_bound = float(bound) if not isinstance(bound, str) else prev_bound

    return avg / 100.0 if total_pct > 0 else None

files = sorted(glob.glob(os.path.join(results_dir, "mix_*_*.json")))
if not files:
    sys.exit(0)

agg = {}
for f in files:
    with open(f) as fh:
        j = json.load(fh)
    for job in j.get("jobs", []):
        m = re.search(r"rwmix(\d+)", f)
        mix = int(m.group(1)) if m else 100
        read = job.get("read", {}); write = job.get("write", {})
        bw_MBps = (read.get("bw", 0) + write.get("bw", 0))/1024.0
        lat_us = extract_latency(job)
        agg.setdefault(mix, []).append((bw_MBps, lat_us))

# aggregate by mix
mixes = sorted(agg.keys(), reverse=True)  # 100R .. 0R
bw_mean=[]; bw_err=[]; lat_mean=[]; lat_err=[]
for m in mixes:
    bws = [x[0] for x in agg[m]]
    lats = [x[1] for x in agg[m]]
    bw_mean.append(mean(bws)); bw_err.append(stdev(bws) if len(bws)>1 else 0.0)
    lat_mean.append(mean(lats)); lat_err.append(stdev(lats) if len(lats)>1 else 0.0)

plt.figure(figsize=(7,5))
plt.errorbar(mixes, bw_mean, marker="o", linestyle="-")
plt.xlabel("Read mix (% reads)")
plt.ylabel("Throughput (MB/s)")
plt.title("Read/Write Mix Sweep — Throughput")
plt.grid(True)
out=os.path.join(figs_dir,"mix_throughput.png")
plt.tight_layout(); plt.savefig(out); plt.close()

plt.figure(figsize=(7,5))
plt.errorbar(mixes, lat_mean, marker="o", linestyle="-")
plt.xlabel("Read mix (% reads)")
plt.ylabel("Average latency (µs)")
plt.title("Read/Write Mix Sweep — Latency")
plt.grid(True)
out=os.path.join(figs_dir,"mix_latency.png")
plt.tight_layout(); plt.savefig(out); plt.close()
