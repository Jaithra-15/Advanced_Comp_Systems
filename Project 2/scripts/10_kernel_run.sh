#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
BIN=./saxpy

# Examples for miss shaping:
#  - Stride = 1 (good locality), 64 (cache line), 4096 (page stride -> TLB pressure)
#  - Random pattern to defeat HW prefetch
for pat in seq random; do
  for stride in 1 64 4096; do
    for T in 1 4; do
      CMD="$BIN --N $((1<<26)) --stride $stride --threads $T --rw $pat --footprint_MB 512"
      echo "Running: $CMD"
      perf stat -x, -e \
        cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,\
dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses \
        $CMD 2>> data/kernel_perf.csv
    done
  done
done
