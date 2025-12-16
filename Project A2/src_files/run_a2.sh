#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# User knobs (env overrides)
# ----------------------------
RUNS="${RUNS:-3}"
QUICK="${QUICK:-0}"
PERF="${PERF:-0}"              # PERF=1 collects only supported counters (no cycles/instructions)
PIN_MHZ="${PIN_MHZ:-2400}"     # used for cycles_est = seconds * PIN_MHZ*1e6
CPUSET="${CPUSET:-0-15}"       # pin process to these CPUs

OUTDIR="results"
OUTCSV="${OUTDIR}/results_a2.csv"
PERF_TMP="${OUTDIR}/perf_tmp.csv"

CXX="${CXX:-g++}"
CXXFLAGS="-O3 -march=native -std=c++17 -fopenmp"

mkdir -p "${OUTDIR}"

echo "[build] ${CXX} ${CXXFLAGS} a2_benchmark.cpp a2_kernels.cpp a2_utils.cpp -o a2_benchmark"
${CXX} ${CXXFLAGS} a2_benchmark.cpp a2_kernels.cpp a2_utils.cpp -o a2_benchmark

# Header
if [[ ! -f "${OUTCSV}" ]]; then
  ./a2_benchmark --header 1 > "${OUTCSV}"
fi

# Best-effort CPU freq pin (may not exist on WSL; still record PIN_MHZ either way)
pin_freq_best_effort() {
  if [[ -d /sys/devices/system/cpu/cpufreq ]]; then
    echo "[freq] attempting to pin frequency to ~${PIN_MHZ} MHz (best effort)"
    # This may require sudo. If it fails, continue.
    sudo bash -c "
      for p in /sys/devices/system/cpu/cpufreq/policy*; do
        [[ -f \$p/scaling_governor ]] && echo performance > \$p/scaling_governor || true
        [[ -f \$p/scaling_min_freq ]] && echo $((PIN_MHZ*1000)) > \$p/scaling_min_freq || true
        [[ -f \$p/scaling_max_freq ]] && echo $((PIN_MHZ*1000)) > \$p/scaling_max_freq || true
      done
    " || true
  else
    echo "[freq] cpufreq sysfs not available (common on WSL). Will still estimate cycles using PIN_MHZ=${PIN_MHZ}."
  fi
}
pin_freq_best_effort

# Threads to sweep
if [[ "${QUICK}" == "1" ]]; then
  THREADS=(1 2 4 8)
else
  THREADS=(1 2 4 8 12 16)
fi

# Helper: run one config, append one row (and only supported perf counters)
run_one() {
  local kernel="$1"
  local variant="$2"
  local m="$3"
  local k="$4"
  local n="$5"
  local density="$6"
  local pattern="$7"
  local layoutB="$8"
  local threads="$9"
  local tileM="${10}"
  local tileN="${11}"
  local tileK="${12}"
  local jblock="${13}"
  local seed="${14}"
  local runid="${15}"

  local cmd=(./a2_benchmark
    --kernel "${kernel}"
    --variant "${variant}"
    --layoutB "${layoutB}"
    --pattern "${pattern}"
    --m "${m}" --k "${k}" --n "${n}"
    --density "${density}"
    --threads "${threads}"
    --tileM "${tileM}" --tileN "${tileN}" --tileK "${tileK}"
    --jblock "${jblock}"
    --seed "${seed}"
    --run "${runid}"
    --freq_mhz "${PIN_MHZ}"
  )

  if [[ "${PERF}" == "1" ]]; then
    # Only counters that work in your environment
    perf stat -x, -e task-clock,context-switches,cpu-migrations,page-faults \
      -o "${PERF_TMP}" -- taskset -c "${CPUSET}" "${cmd[@]}" > "${OUTDIR}/row_tmp.txt" || true

    # Extract perf values (first column)
    local task_clock ctx mig pf
    task_clock="$(awk -F, '/task-clock/ {gsub(/ /,"",$1); print $1; exit}' "${PERF_TMP}")"
    ctx="$(awk -F, '/context-switches/ {gsub(/ /,"",$1); print $1; exit}' "${PERF_TMP}")"
    mig="$(awk -F, '/cpu-migrations/ {gsub(/ /,"",$1); print $1; exit}' "${PERF_TMP}")"
    pf="$(awk -F, '/page-faults/ {gsub(/ /,"",$1); print $1; exit}' "${PERF_TMP}")"
    task_clock="${task_clock:-0}"
    ctx="${ctx:-0}"
    mig="${mig:-0}"
    pf="${pf:-0}"

    # Re-run once without perf to print row with correct perf columns populated
    # (cheap relative to full sweep; avoids merging columns by hand)
    taskset -c "${CPUSET}" "${cmd[@]}" \
      --perf_task_clock_ms "${task_clock}" \
      --perf_context_switches "${ctx}" \
      --perf_cpu_migrations "${mig}" \
      --perf_page_faults "${pf}" >> "${OUTCSV}"
  else
    taskset -c "${CPUSET}" "${cmd[@]}" >> "${OUTCSV}"
  fi
}

# ----------------------------
# Always include STREAM once
# ----------------------------
echo "[run] stream bandwidth"
run_one stream scalar 1 1 1 1.0 uniform row 1 0 0 0 0 123 0

# -----------------------------------------
# Figure 1: GEMM scaling (scalar + simd)
# -----------------------------------------
echo "[run] GEMM scaling (scalar+simd) across threads"
for t in "${THREADS[@]}"; do
  for r in $(seq 1 "${RUNS}"); do
    run_one gemm scalar 1536 1536 1536 1.0 uniform row "${t}" 64 128 64 128 100 "$r"
    run_one gemm simd   1536 1536 1536 1.0 uniform row "${t}" 64 128 64 128 100 "$r"
  done
done

# -----------------------------------------
# Figure 2: CSR-SpMM scaling (scalar + simd)
# -----------------------------------------
echo "[run] CSR-SpMM scaling (scalar+simd) across threads"
for t in "${THREADS[@]}"; do
  for r in $(seq 1 "${RUNS}"); do
    run_one spmm_csr scalar 2048 2048 512 0.01 uniform row "${t}" 64 128 64 128 200 "$r"
    run_one spmm_csr simd   2048 2048 512 0.01 uniform row "${t}" 64 128 64 128 200 "$r"
  done
done

# ------------------------------------------------
# Extra sweeps for break-even + working-set plots
# (keep simd only to control runtime)
# ------------------------------------------------
if [[ "${QUICK}" == "0" ]]; then
  echo "[run] density break-even sweep (simd)"
  DENSITIES=(0.001 0.002 0.005 0.01 0.02 0.05 0.10 0.20 0.50)
  for d in "${DENSITIES[@]}"; do
    for r in $(seq 1 "${RUNS}"); do
      run_one spmm_csr simd 2048 2048 512 "${d}" uniform row 8 64 128 64 128 300 "$r"
      run_one gemm     simd 2048 2048 512 1.0    uniform row 8 64 128 64 128 300 "$r"
    done
  done

  echo "[run] working-set size sweep (simd)"
  SIZES=(256 512 768 1024 1536 2048 3072)
  for s in "${SIZES[@]}"; do
    for r in $(seq 1 "${RUNS}"); do
      run_one gemm     simd "${s}" "${s}" "${s}" 1.0  uniform row 8 64 128 64 128 400 "$r"
      run_one spmm_csr simd "${s}" "${s}" 256    0.01 uniform row 8 64 128 64 128 400 "$r"
    done
  done
fi

echo "[run] done -> ${OUTCSV}"
echo "[plot] python3 plot_a2.py --csv ${OUTCSV} --outdir ${OUTDIR}"

