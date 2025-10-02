#!/usr/bin/env bash
set -euo pipefail

# ---------------- Config ----------------
SRC_FILE="page_miss.cpp"
EXE_FILE="page_miss"
OUT_CSV="page_miss_results.csv"

# ---------------- Compile ----------------
g++ -O3 -march=native -std=c++17 -o "$EXE_FILE" "$SRC_FILE"

# ---------------- Prepare CSV ----------------
echo "pages,ns_per_access" > "$OUT_CSV"

# ---------------- Run Benchmark ----------------
while read -r pages ns; do
    echo "$pages,$ns" >> "$OUT_CSV"
done < <(./"$EXE_FILE")

echo "Run complete. Results saved to $OUT_CSV"
