#!/usr/bin/env python3
import sys, glob, os, re, json
import matplotlib.pyplot as plt
from statistics import mean

# Input/output directories
results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
figs_dir = sys.argv[2] if len(sys.argv) > 2 else "figs"
os.makedirs(figs_dir, exist_ok=True)

def extract_percentiles(job):
    """Extract p50, p95, p99, p99.9 from fio output (read/write). Returns dict in µs."""
    def get_perc(d):
        return {
            "p50": float(d.get("50.000000", 0.0)) / 1000.0,
            "p95": float(d.get("95.000000", 0.0)) / 1000.0,
            "p99": float(d.get("99.000000", 0.0)) / 1000.0,
            "p99.9": float(d.get("99.900000", 0.0)) / 1000.0,
        }

    if "read" in job and "clat_ns" in job["read"] and "percentile" in job["read"]["clat_ns"]:
        return get_perc(job["read"]["clat_ns"]["percentile"])
    if "write" in job and "clat_ns" in job["write"] and "percentile" in job["write"]["clat_ns"]:
        return get_perc(job["write"]["clat_ns"]["percentile"])
    return None

# Collect JSON files (pattern qd_rand_4k_qdx_ry.json)
files = sorted(glob.glob(os.path.join(results_dir, "qd_rand_4k_qd*_r*.json")))
if not files:
    sys.exit(0)

tail_data = {}
for f in files:
    with open(f) as fh:
        j = json.load(fh)
    for job in j.get("jobs", []):
        # parse QD and run from filename
        m = re.search(r"qd_rand_4k_qd(\d+)_r(\d+)\.json", f)
        if not m:
            continue
        qd = int(m.group(1))
        run = int(m.group(2))

        percs = extract_percentiles(job)
        if percs:
            tail_data.setdefault(qd, []).append(percs)

# Aggregate across runs
qds = sorted(tail_data.keys())
p50_mean=[]; p95_mean=[]; p99_mean=[]; p999_mean=[]
for q in qds:
    p50s = [d["p50"] for d in tail_data[q] if d["p50"] > 0]
    p95s = [d["p95"] for d in tail_data[q] if d["p95"] > 0]
    p99s = [d["p99"] for d in tail_data[q] if d["p99"] > 0]
    p999s = [d["p99.9"] for d in tail_data[q] if d["p99.9"] > 0]

    p50_mean.append(mean(p50s) if p50s else None)
    p95_mean.append(mean(p95s) if p95s else None)
    p99_mean.append(mean(p99s) if p99s else None)
    p999_mean.append(mean(p999s) if p999s else None)

# Plot Tail Latency vs QD
plt.figure(figsize=(7,5))
plt.plot(qds, p50_mean, marker="o", label="p50")
plt.plot(qds, p95_mean, marker="o", label="p95")
plt.plot(qds, p99_mean, marker="o", label="p99")
plt.plot(qds, p999_mean, marker="o", label="p99.9")

# Vertical line at QD=8
plt.axvline(x=8, color="gray", linestyle="--")

plt.yscale("log")
plt.xlabel("Queue Depth")
plt.ylabel("Latency (µs, log scale)")
plt.title("Tail Latency Characterization (Random 4K)")
plt.grid(True, which="both", ls="--")
plt.legend()
out = os.path.join(figs_dir, "tail_latency.png")
plt.tight_layout()
plt.savefig(out)
plt.close()
