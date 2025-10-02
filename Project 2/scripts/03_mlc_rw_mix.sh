#!/usr/bin/env bash
set -euo pipefail

MLC=${MLC:-./mlc}
OUT=data/mlc_latency_rw.csv
mkdir -p data

# Correct header
echo "rw_mix,size,latency_ns,bandwidth_MBps" > "$OUT"

# R/W mixes: read-only, write-only, 70R/30W, 50R/50W
declare -A MIX_ARGS
MIX_ARGS=(
  ["allreads"]="--loaded_latency -r"
  ["allwrites"]="--loaded_latency -w6"
  ["r70w30"]="--loaded_latency -w3"
  ["r50w50"]="--loaded_latency -w5"
)

SIZES=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
       8192 16384 32768 65536 131072 262144 524288)

# Force deterministic order
MIX_ORDER=("allreads" "allwrites" "r70w30" "r50w50")

for mix in "${MIX_ORDER[@]}"; do
  echo ">>> Running $mix tests..."
  for size in "${SIZES[@]}"; do
    echo ">>> Running $size tests..."

    # Run MLC
    echo ">>> Running: $MLC ${MIX_ARGS[$mix]} -b${size}"
    out=$(sudo $MLC ${MIX_ARGS[$mix]} -b${size})

    echo "$out" >> "data/output.txt"


    echo ">>>Extracter"

    # Add each inject manually
    ns=$(echo "$out" | grep "00000" | awk '{print $2}'); bw=$(echo "$out" | grep "00000" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00002" | awk '{print $2}'); bw=$(echo "$out" | grep "00002" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00008" | awk '{print $2}'); bw=$(echo "$out" | grep "00008" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00015" | awk '{print $2}'); bw=$(echo "$out" | grep "00015" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00050" | awk '{print $2}'); bw=$(echo "$out" | grep "00050" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00100" | awk '{print $2}'); bw=$(echo "$out" | grep "00100" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00200" | awk '{print $2}'); bw=$(echo "$out" | grep "00200" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00300" | awk '{print $2}'); bw=$(echo "$out" | grep "00300" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00400" | awk '{print $2}'); bw=$(echo "$out" | grep "00400" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00500" | awk '{print $2}'); bw=$(echo "$out" | grep "00500" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "00700" | awk '{print $2}'); bw=$(echo "$out" | grep "00700" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "01000" | awk '{print $2}'); bw=$(echo "$out" | grep "01000" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "01300" | awk '{print $2}'); bw=$(echo "$out" | grep "01300" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "01700" | awk '{print $2}'); bw=$(echo "$out" | grep "01700" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "02500" | awk '{print $2}'); bw=$(echo "$out" | grep "02500" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "03500" | awk '{print $2}'); bw=$(echo "$out" | grep "03500" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "05000" | awk '{print $2}'); bw=$(echo "$out" | grep "05000" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "09000" | awk '{print $2}'); bw=$(echo "$out" | grep "09000" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"
    ns=$(echo "$out" | grep "20000" | awk '{print $2}'); bw=$(echo "$out" | grep "20000" | awk '{print $3}'); echo "$mix,$size,$ns,$bw" >> "$OUT"

  done
done

echo "Done! Results in $OUT"
