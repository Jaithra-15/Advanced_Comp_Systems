#!/usr/bin/env python3
import sys, glob, os, re, json, math
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

files = sorted(glob.glob(os.path.join(results_dir, "qd_*_*.json")))
if not files:
    sys.exit(0)

by_qd = {}
for f in files:
    with open(f) as fh:
        j = json.load(fh)
    for job in j.get("jobs", []):
        m = re.search(r"_qd(\d+)_", f)
        qd = int(m.group(1)) if m else 1
        read = job.get("read", {}); write = job.get("write", {})
        bw_MBps = (read.get("bw", 0) + write.get("bw", 0)) / 1024.0

        # --- latency handling (normalized to µs) ---
        lat_us = extract_latency(job)
        

        by_qd.setdefault(qd, []).append((bw_MBps, lat_us))

qds = sorted(by_qd)
bw_mean=[]; bw_err=[]; lat_mean=[]; lat_err=[]
for q in qds:
    bws = [x[0] for x in by_qd[q]]
    lats = [x[1] for x in by_qd[q]]
    bw_mean.append(mean(bws)); bw_err.append(stdev(bws) if len(bws) > 1 else 0.0)
    lat_mean.append(mean(lats)); lat_err.append(stdev(lats) if len(lats) > 1 else 0.0)

# Throughput-latency curve
# Throughput-latency curve (bandwidth on x, latency on y)
plt.figure(figsize=(7,5))
plt.errorbar(bw_mean, lat_mean, xerr=bw_err, yerr=lat_err,
             marker="o", linestyle="-")

# Label each point with its queue depth
for i, q in enumerate(qds):
    plt.annotate(f"QD{q}", (bw_mean[i], lat_mean[i]),
                 textcoords="offset points", xytext=(5,5), fontsize=8)

plt.xlabel("Throughput (MB/s)")
plt.ylabel("Average latency (µs)")
plt.title("Latency vs Throughput (QD Sweep)")
plt.grid(True)

out = os.path.join(figs_dir, "qd_tradeoff_curve.png")
plt.tight_layout()
plt.savefig(out)
plt.close()


# Knee detection (Kneedle-like on normalized curve vs QD)
lx = [(x - min(lat_mean)) / (max(lat_mean) - min(lat_mean) + 1e-9) for x in lat_mean]
ly = [(y - min(bw_mean)) / (max(bw_mean) - min(bw_mean) + 1e-9) for y in bw_mean]

curvatures = []
for i in range(1, len(qds)-1):
    x1,y1 = lx[i-1], ly[i-1]
    x2,y2 = lx[i], ly[i]
    x3,y3 = lx[i+1], ly[i+1]
    area = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)) / 2.0
    curvatures.append((area, i))

if curvatures:
    knee_idx = max(curvatures)[1]
    knee_qd = qds[knee_idx]
    with open(os.path.join(figs_dir, "qd_knee.txt"), "w") as fh:
        fh.write(f"KNEE_QD={knee_qd}\n")
