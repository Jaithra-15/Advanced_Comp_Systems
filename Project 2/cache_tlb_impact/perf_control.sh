#!/usr/bin/env bash
set -euo pipefail

APP=${APP:-./your_bench}   # your workload (take size/stride/pattern args)
OUT=${OUT:-perf_cache_impact.csv}
echo "bytes,stride,pattern,sec,cycles,instr,ipc,L1_ld,L1_miss,LLC_ld,LLC_miss,br_miss,frontend_stall,backend_stall" >"$OUT"

sizes=( 64K 512K 2M 8M 32M 128M 512M )
strides=(64 256 4096)          # 64B, 256B, 4KB
patterns=(seq rand)

for B in "${sizes[@]}"; do
  for S in "${strides[@]}"; do
    for P in "${patterns[@]}"; do
      # 5 repeats to reduce noise
      perf stat -x, -r 5 \
        -e cycles,instructions,branches,branch-misses, \
           L1-dcache-loads,L1-dcache-load-misses, \
           LLC-loads,LLC-load-misses, \
           frontend_cycles_idle,backend_cycles_idle \
        $APP --bytes "$B" --stride "$S" --pattern "$P" >/dev/null 2>tmp.csv

      # perf -x, produces CSV (one line per metric per repeat + summary)
      # Extract the 'summary' (the last group after -r) with awk
      secs=$(awk -F, '/seconds time elapsed/{t=$1} END{print t}' tmp.csv)
      cyc=$(awk -F, '/ cycles /{c=$1} END{print c}' tmp.csv)
      ins=$(awk -F, '/ instructions /{c=$1} END{print c}' tmp.csv)
      ipc=$(awk -v c="$cyc" -v i="$ins" 'BEGIN{printf "%.4f", (c>0? i/c : 0)}')

      L1ld=$(awk -F, '/ L1-dcache-loads /{c=$1} END{print c}' tmp.csv)
      L1ms=$(awk -F, '/ L1-dcache-load-misses /{c=$1} END{print c}' tmp.csv)
      LLCl=$(awk -F, '/ LLC-loads /{c=$1} END{print c}' tmp.csv)
      LLCm=$(awk -F, '/ LLC-load-misses /{c=$1} END{print c}' tmp.csv)
      brm=$(awk -F, '/ branch-misses /{c=$1} END{print c}' tmp.csv)
      fst=$(awk -F, '/ frontend_cycles_idle /{c=$1} END{print c}' tmp.csv)
      bst=$(awk -F, '/ backend_cycles_idle /{c=$1} END{print c}' tmp.csv)

      echo "$B,$S,$P,$secs,$cyc,$ins,$ipc,$L1ld,$L1ms,$LLCl,$LLCm,$brm,$fst,$bst" >>"$OUT"
    done
  done
done
