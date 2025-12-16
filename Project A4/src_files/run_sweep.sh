#!/bin/bash
set -euo pipefail

# =======================
# User-configurable knobs
# =======================
FREQ_GHZ="1.5"
FREQ_HZ="$(python3 - <<'PY'
freq_ghz = float("1.5")
print(int(freq_ghz * 1e9))
PY
)"

THREADS=("1" "2" "4" "8" "16")
KEYS=("10000" "100000" "1000000")
MODES=("lookup" "insert" "mixed")
IMPLS=("coarse" "fine")

OPS=2000000
REPS=5

OUT="results.csv"

# =======================
# Helpers
# =======================
say() { echo -e "[$(date '+%F %T')] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }
}

# =======================
# Preconditions
# =======================
need_cmd gcc
need_cmd python3

say "===================================================="
say "Project A4 sweep starting"
say "Target CPU frequency: ${FREQ_GHZ} GHz (${FREQ_HZ} Hz)"
say "OPS per run: ${OPS}, REPS per config: ${REPS}"
say "Output CSV: ${OUT}"
say "Working dir: $(pwd)"
say "===================================================="

# =======================
# 1) Pin CPU frequency (best-effort)
# =======================
say "Pinning CPU frequency (best-effort). This may require sudo."
if command -v cpupower >/dev/null 2>&1; then
  say "Using cpupower..."
  sudo -n true 2>/dev/null || say "Note: sudo may prompt for password."
  sudo cpupower frequency-set -g performance || true
  sudo cpupower frequency-set -f "${FREQ_GHZ}GHz" || true
elif command -v cpufreq-set >/dev/null 2>&1; then
  say "Using cpufreq-set..."
  sudo -n true 2>/dev/null || say "Note: sudo may prompt for password."
  sudo cpufreq-set -g performance || true
  sudo cpufreq-set -f "${FREQ_GHZ}GHz" || true
else
  say "cpupower/cpufreq-set not found. Skipping frequency pin step."
  say "If you already pinned frequency manually, that's OK."
fi

# Log current governor/frequency if available
if [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
  GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true)
  CUR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq || true)
  MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq || true)
  MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq || true)
  say "CPU0 governor=${GOV}, cur_freq=${CUR} kHz, min=${MIN} kHz, max=${MAX} kHz"
else
  say "Could not read cpufreq sysfs info; continuing."
fi

# =======================
# 2) Compile
# =======================
say "Compiling benchmark..."
gcc -O2 -pthread -o benchmark benchmark.c hashtable.c
say "Build OK: ./benchmark"

# =======================
# 3) Write CSV header
# =======================
echo "run_id,impl,mode,keys,threads,ops,prefill,seconds,throughput" > "${OUT}"

# =======================
# 4) Run sweep
# =======================
TOTAL=0
for k in "${KEYS[@]}"; do
  for m in "${MODES[@]}"; do
    for i in "${IMPLS[@]}"; do
      for t in "${THREADS[@]}"; do
        TOTAL=$((TOTAL + REPS))
      done
    done
  done
done

say "Total benchmark runs: ${TOTAL}"
RUN=0
START=$(date +%s)

for k in "${KEYS[@]}"; do
  for m in "${MODES[@]}"; do
    for i in "${IMPLS[@]}"; do
      for t in "${THREADS[@]}"; do
        for r in $(seq 1 "${REPS}"); do
          RUN=$((RUN + 1))

          prefill="${k}"
          if [[ "${m}" == "insert" ]]; then
            prefill="0"
          fi

          say "Run ${RUN}/${TOTAL}: impl=${i} mode=${m} keys=${k} threads=${t} rep=${r}/${REPS} prefill=${prefill}"

          ./benchmark \
            --runid "${RUN}" \
            --impl "${i}" \
            --mode "${m}" \
            --threads "${t}" \
            --keys "${k}" \
            --ops "${OPS}" \
            --prefill "${prefill}" \
            >> "${OUT}"

          if (( RUN % 25 == 0 )); then
            NOW=$(date +%s)
            ELAPSED=$((NOW - START))
            say "---- Progress: ${RUN}/${TOTAL}, elapsed ${ELAPSED}s"
          fi
        done
      done
    done
  done
done

END=$(date +%s)
say "Sweep complete. Elapsed: $((END - START))s"
say "Wrote ${OUT}"

# =======================
# 5) Plot (derived cycles from time + pinned frequency)
# =======================
if [[ -f plot_results.py ]]; then
  say "Generating plots with derived cycles using FREQ_HZ=${FREQ_HZ}..."
  python3 plot_results.py "${OUT}" "${FREQ_HZ}"
  say "Plots generated."
else
  say "plot_results.py not found in this directory. Skipping plot step."
fi

say "DONE."

