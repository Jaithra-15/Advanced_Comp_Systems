#!/usr/bin/env bash
set -euo pipefail
MLC=${MLC:-./mlc}
OUT=data/mlc_loaded_latency.csv
echo "threads,throughput_MBps,avg_latency_ns" > $OUT

# Sweep concurrency: 1,2,4,8,... (adjust to core count)
for t in 1 2 4 8; do
  sudo $MLC --loaded_latency -t$t \
    | awk -v th=$t '/^TOTAL/ {tp=$2} /Avg Latency/ {lat=$3} END {printf("%d,%.1f,%.1f\n",th,tp,lat)}' \
    >> $OUT
done
