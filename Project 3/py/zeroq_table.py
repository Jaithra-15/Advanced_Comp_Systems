#!/usr/bin/env python3
import json, pathlib, sys
import matplotlib.pyplot as plt

results_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("../results")
json_files = sorted(results_dir.glob("zeroq_*.json"))
if not json_files:
    sys.exit(f"No JSON files found in {results_dir}")

def compute_avg_latency_ms(job):
    """
    Compute weighted average latency from fio histogram
    (latency_ns, latency_us, latency_ms if available).
    Returns average latency in milliseconds.
    """
    buckets = {}

    # Convert everything into milliseconds
    if "latency_ns" in job:
        # ns → ms
        buckets.update({k: v for k, v in job["latency_ns"].items()})
        buckets = {float(k) / 1_000_000.0: v for k, v in buckets.items()}

    if "latency_us" in job:
        # µs → ms
        for k, v in job["latency_us"].items():
            buckets[float(k) / 1000.0] = v

    if "latency_ms" in job:
        for k, v in job["latency_ms"].items():
            if k.startswith(">="):
                buckets[k] = v  # keep as-is, handle later
            else:
                buckets[float(k)] = v

    if not buckets:
        return None

    # Now compute weighted average
    avg = 0.0
    total_pct = 0.0
    prev_bound = 0.0

    for bound, pct in sorted(buckets.items(), key=lambda x: (float(x[0].strip(">=")) if isinstance(x[0], str) else x[0])):
        pct = float(pct)
        if pct <= 0.0:
            continue

        if isinstance(bound, str) and bound.startswith(">="):
            # e.g. ">=2000" ms → assume midpoint at 1.5 × lower bound
            lower = float(bound[2:])
            midpoint = lower * 1.5
        else:
            midpoint = (prev_bound + float(bound)) / 2.0

        avg += midpoint * pct
        total_pct += pct
        prev_bound = float(bound) if not isinstance(bound, str) else prev_bound

    return avg / 100.0 if total_pct > 0 else None


def extract_percentiles(job):
    """Extract 95th and 99th percentiles if present."""
    if "percentile" in job["read"]["clat_ns"]:
        return (

            job["read"]["clat_ns"]["percentile"].get("95.000000", 0.0) / 1000000.0,
            job["read"]["clat_ns"]["percentile"].get("99.000000", 0.0) / 1000000.0,
        )
    if "percentile" in job["write"]["clat_ns"]:
        return (
            job["write"]["clat_ns"]["percentile"].get("95.000000", 0.0) / 1000000.0,
            job["write"]["clat_ns"]["percentile"].get("99.000000", 0.0) / 1000000.0,
        )
    return (0.0, 0.0)

rows = []
for jf in json_files:
    try:
        data = json.load(open(jf))
    except Exception as e:
        print(f"Skipping {jf}, invalid JSON: {e}")
        continue

    for job in data.get("jobs", []):
        name = job["jobname"]

        # Compute average latency from latency_ms
        avg_ms = compute_avg_latency_ms(job)

        # Extract percentiles
        p95, p99 = extract_percentiles(job)

        # IOPS and bandwidth
        read, write = job.get("read", {}), job.get("write", {})
        iops = read.get("iops", 0.0) + write.get("iops", 0.0)
        bw_kbps = read.get("bw", 0) + write.get("bw", 0)
        bw_MBps = bw_kbps / 1024.0

        # Friendly label
        label = name.replace("_", " ")
        label = (label
                 .replace("randread 4k", "4KiB Random Read")
                 .replace("randwrite 4k", "4KiB Random Write")
                 .replace("read 128k", "128KiB Seq Read")
                 .replace("write 128k", "128KiB Seq Write"))

        rows.append((label, avg_ms, p95, p99, iops, bw_MBps))

# Markdown table
print("| Test | Avg Lat (ms, from histogram) | p95 (µs) | p99 (µs) | IOPS | Bandwidth (MB/s) |")
print("|------|-----------------------------:|---------:|---------:|-----:|-----------------:|")
for r in rows:
    print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.0f} | {r[5]:.1f} |")

# --- Plot ---
labels = [r[0] for r in rows]
avg_vals = [r[1] for r in rows]
iops_vals = [r[4] for r in rows]
bw_vals = [r[5] for r in rows]

fig, ax1 = plt.subplots(figsize=(8, 5))
color = 'tab:blue'
ax1.set_xlabel('Test')
ax1.set_ylabel('Bandwidth (MB/s)', color=color)
ax1.bar(labels, bw_vals, color=color, alpha=0.6)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Latency (ms)', color=color)
ax2.plot(labels, avg_vals, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("ZeroQ Results: Bandwidth vs Latency")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()
