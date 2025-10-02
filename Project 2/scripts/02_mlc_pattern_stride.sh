#!/usr/bin/env bash
set -euo pipefail

MLC=${MLC:-./mlc}
OUT=data/mlc_patterns.csv
mkdir -p data

# Header
echo "pattern,size,latency_ns,bandwidth_MBps" > "$OUT"

# Block sizes (you can adjust this list if needed)
SIZES=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
       8192 16384 32768 65536 131072 262144 524288)

# Injection delays to extract
INJECTS=(00000 00002 00008 00015 00050 00100 00200 00300 \
         00400 00500 00700 01000 01300 01700 02500 03500 \
         05000 09000 20000)

for size in "${SIZES[@]}"; do
  for pattern in seq rand; do
    if [[ "$pattern" == "seq" ]]; then
      cmd="$MLC --loaded_latency -b${size}"
    else
      cmd="$MLC --loaded_latency -r -U -b${size}"
    fi

    echo ">>> Running $pattern, size=$size ..."
    out=$(sudo $cmd)
    echo "$out" >> data/output_seq_rand.txt

    for inj in "${INJECTS[@]}"; do
      ns=$(echo "$out" | awk -v inj=$inj '$1==inj {print $2}')
      bw=$(echo "$out" | awk -v inj=$inj '$1==inj {print $3}')
      if [[ -n "$ns" && -n "$bw" ]]; then
        echo "$pattern,$size,$ns,$bw" >> "$OUT"
      fi
    done
  done
done

echo "✅ Done! Results in $OUT"
