#!/usr/bin/env bash
set -euo pipefail

OUT=perf_counters.csv
echo "mode,kernel,dtype,N,access,stride,misaligned,tail,\
instructions,cycles,CPI,\
L1_loads,L1_miss,L1_miss_rate,\
LLC_loads,LLC_miss,LLC_miss_rate,\
dTLB_loads,dTLB_miss,dTLB_miss_rate" > "$OUT"

for access in unit strided gather; do
  for stride in 1 2 4 8 16 32 64 128 256 512 1024; do
    perf stat -x, -e \
      instructions,cycles,cache-references,cache-misses,\
      L1-dcache-loads,L1-dcache-load-misses,\
      LLC-loads,LLC-load-misses,\
      dTLB-loads,dTLB-load-misses \
      ./your_bench --access=$access --stride=$stride --trials=30 --cpu_ghz=2.6 \
      2> tmp.perf

    # Extract numbers with defaults
    get_val() { awk -F, -v key="$1" '$3==key{print $1}' tmp.perf | head -n1; }
    INSTR=$(get_val "instructions");   INSTR=${INSTR:-0}
    CYC=$(get_val "cycles");           CYC=${CYC:-0}
    L1L=$(get_val "L1-dcache-loads");  L1L=${L1L:-0}
    L1M=$(get_val "L1-dcache-load-misses"); L1M=${L1M:-0}
    LLCL=$(get_val "LLC-loads");       LLCL=${LLCL:-0}
    LLCM=$(get_val "LLC-load-misses"); LLCM=${LLCM:-0}
    DTLBL=$(get_val "dTLB-loads");     DTLBL=${DTLBL:-0}
    DTLBM=$(get_val "dTLB-load-misses"); DTLBM=${DTLBM:-0}

    # Derived metrics
    CPI=0
    if [[ "$INSTR" -gt 0 ]]; then
      CPI=$(awk -v c="$CYC" -v i="$INSTR" 'BEGIN{printf "%.4f", c/i}')
    fi

    L1_RATE=0
    if [[ "$L1L" -gt 0 ]]; then
      L1_RATE=$(awk -v m="$L1M" -v l="$L1L" 'BEGIN{printf "%.6f", m/l}')
    fi

    LLC_RATE=0
    if [[ "$LLCL" -gt 0 ]]; then
      LLC_RATE=$(awk -v m="$LLCM" -v l="$LLCL" 'BEGIN{printf "%.6f", m/l}')
    fi

    DTLB_RATE=0
    if [[ "$DTLBL" -gt 0 ]]; then
      DTLB_RATE=$(awk -v m="$DTLBM" -v l="$DTLBL" 'BEGIN{printf "%.6f", m/l}')
    fi

    # Append row
    echo "SIMD,saxpy,f32,1048576,$access,$stride,0,1,\
$INSTR,$CYC,$CPI,\
$L1L,$L1M,$L1_RATE,\
$LLCL,$LLCM,$LLC_RATE,\
$DTLBL,$DTLBM,$DTLB_RATE" >> "$OUT"
  done
done
