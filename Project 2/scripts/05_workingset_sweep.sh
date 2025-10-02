#!/usr/bin/env bash
set -euo pipefail
MLC=${MLC:-./mlc}
OUT=data/mlc_workingset_latency.csv
echo "size_MB,latency_ns" > $OUT

# Grow pointer-chase footprint to cross L1/L2/L3/DRAM; step sizes tuned to your caches
for sz in 0.032 0.064 0.128 0.256 0.5 1 2 4 8 16 32 64 128 256; do
  sudo $MLC --idle_latency -b${sz}M \
    | awk -v s=$sz '/^Memory latency/ {print s","$3}' >> $OUT
done
  
