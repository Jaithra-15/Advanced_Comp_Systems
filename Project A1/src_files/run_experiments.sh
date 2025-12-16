#!/usr/bin/env bash
# run_experiments.sh
# Compile benchmarks and run experiments for Features 1–4.

set -euo pipefail

###############################################################################
# Environment control: fix CPU frequency, remind to minimize background load
###############################################################################

echo "=== Environment preparation ==="

# Try to set performance governor (requires cpupower + sudo)
if command -v cpupower >/dev/null 2>&1; then
  echo "Attempting to set CPU governor to 'performance' (may prompt for sudo)..."
  if sudo cpupower frequency-set -g performance; then
    echo "CPU governor set to performance."
  else
    echo "WARNING: could not set governor to performance (check permissions / cpupower setup)."
  fi
else
  echo "cpupower not found. Consider installing it and setting governor to performance manually."
fi

echo
echo "Please close heavy background applications (browsers, IDEs, video players) before continuing."
echo "Press Ctrl+C to abort, or waiting 5 seconds to continue..."
sleep 5

###############################################################################
# Build benchmarks
###############################################################################

mkdir -p build results

echo "Compiling benchmarks..."
g++ -O3 -march=native -std=c++17 cpu_burn.cpp     -o build/cpu_burn
g++ -O3 -march=native -std=c++17 mem_scan.cpp     -o build/mem_scan
g++ -O3 -march=native -std=c++17 mem_pattern.cpp  -o build/mem_pattern

# Common perf event set:
# Some (cycles, cache-*, dTLB-*) may show "<not supported>" on this machine.
PERF_EVENTS="task-clock,cycles,context-switches,cpu-migrations,dTLB-loads,dTLB-load-misses,cache-references,cache-misses"

###############################################################################
# Helpers
###############################################################################

# For cpu_burn: parse iterations + runtime
# CSV format: config,run,iters,runtime_seconds
run_cpu_burn_with_perf() {
  local config="$1"; shift
  local run="$1"; shift
  local csv="$1"; shift
  local log="$1"; shift
  local cmd=( "$@" )

  # Run perf; program prints cpu_burn summary + RUNTIME_SECONDS to stdout.
  local output
  output=$(perf stat -e "$PERF_EVENTS" -- "${cmd[@]}" 2>> "$log")

  # Parse iterations from the "cpu_burn done: iters=..." line
  local iters
  iters=$(echo "$output" | awk '
    /cpu_burn done:/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^iters=/) {
          gsub("iters=", "", $i);
          print $i;
          exit;
        }
      }
    }')

  # Parse runtime from "RUNTIME_SECONDS <value>" line
  local runtime
  runtime=$(echo "$output" | awk '/^RUNTIME_SECONDS/ {print $2}')

  if [[ -z "$runtime" ]]; then
    echo "WARNING: no RUNTIME_SECONDS found for config=$config run=$run" >&2
  fi
  if [[ -z "$iters" ]]; then
    echo "WARNING: no iters found for config=$config run=$run" >&2
  fi

  echo "${config},${run},${iters},${runtime}" >> "$csv"
}

# Generic runner (for mem_scan, mem_pattern etc.)
# extra_cols: e.g. "pattern,thp_mode" or "pattern,stride"
# CSV columns must already be created with headers before use.
run_with_perf_generic() {
  local extra_cols="$1"; shift
  local run="$1"; shift
  local csv="$1"; shift
  local log="$1"; shift
  local cmd=( "$@" )

  local output
  output=$(perf stat -e "$PERF_EVENTS" -- "${cmd[@]}" 2>> "$log")

  local runtime
  runtime=$(echo "$output" | awk '/^RUNTIME_SECONDS/ {print $2}')

  if [[ -z "$runtime" ]]; then
    echo "WARNING: no RUNTIME_SECONDS found for ${extra_cols} run=$run" >&2
  fi

  echo "${extra_cols},${run},${runtime}" >> "$csv"
}

###############################################################################
# Feature 1: CPU affinity and scheduler
###############################################################################

F1_CSV="results/feature1_affinity.csv"
F1_LOG="results/feature1_affinity_perf.log"
echo "config,run,iters,runtime_seconds" > "$F1_CSV"
: > "$F1_LOG"

echo "=== Running Feature 1 (CPU affinity) ==="

for run in 1 2 3 4 5; do
  echo "  Run $run: single_no_affinity"
  run_cpu_burn_with_perf "single_no_affinity" "$run" "$F1_CSV" "$F1_LOG" \
    ./build/cpu_burn 5

  echo "  Run $run: single_taskset_0"
  run_cpu_burn_with_perf "single_taskset_0" "$run" "$F1_CSV" "$F1_LOG" \
    taskset -c 0 ./build/cpu_burn 5

  echo "  Run $run: two_no_affinity (run1 + run2 concurrent)"
  run_cpu_burn_with_perf "two_no_affinity_run1" "$run" "$F1_CSV" "$F1_LOG" \
    ./build/cpu_burn 5 &
  pid1=$!
  run_cpu_burn_with_perf "two_no_affinity_run2" "$run" "$F1_CSV" "$F1_LOG" \
    ./build/cpu_burn 5 &
  pid2=$!
  wait "$pid1" "$pid2"

  echo "  Run $run: two_same_core (CPU0 both)"
  run_cpu_burn_with_perf "two_same_core_run1" "$run" "$F1_CSV" "$F1_LOG" \
    taskset -c 0 ./build/cpu_burn 5 &
  pid3=$!
  run_cpu_burn_with_perf "two_same_core_run2" "$run" "$F1_CSV" "$F1_LOG" \
    taskset -c 0 ./build/cpu_burn 5 &
  pid4=$!
  wait "$pid3" "$pid4"

  echo "  Run $run: two_diff_core (CPU0 + CPU2)"
  run_cpu_burn_with_perf "two_diff_core_run1" "$run" "$F1_CSV" "$F1_LOG" \
    taskset -c 0 ./build/cpu_burn 5 &
  pid5=$!
  run_cpu_burn_with_perf "two_diff_core_run2" "$run" "$F1_CSV" "$F1_LOG" \
    taskset -c 2 ./build/cpu_burn 5 &
  pid6=$!
  wait "$pid5" "$pid6"
done

###############################################################################
# Feature 2: THP on/off (mem_scan)
###############################################################################

F2_CSV="results/feature2_thp.csv"
F2_LOG="results/feature2_thp_perf.log"
echo "pattern,thp_mode,run,runtime_seconds" > "$F2_CSV"
: > "$F2_LOG"

echo "=== Running Feature 2 (THP) ==="

patterns=("seq" "stride" "rand")
strides=(1 64 1)  # seq->1, stride->64, rand->1 (ignored in code for rand)

for i in "${!patterns[@]}"; do
  pat=${patterns[$i]}
  str=${strides[$i]}

  for thp in never always; do
    echo "  THP=$thp pattern=$pat"
    echo "$thp" | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null || \
      echo "WARNING: could not set THP mode to $thp"

    for run in 1 2 3; do
      extra="${pat},${thp}"
      run_with_perf_generic "$extra" "$run" "$F2_CSV" "$F2_LOG" \
        ./build/mem_scan --size-mb 2048 --pattern "$pat" --stride "$str" --repeats 3
    done
  done
done

###############################################################################
# Feature 3: SMT interference (cpu_burn again)
###############################################################################

F3_CSV="results/feature3_smt.csv"
F3_LOG="results/feature3_smt_perf.log"
echo "scenario,thread,run,iters,runtime_seconds" > "$F3_CSV"
: > "$F3_LOG"

echo "=== Running Feature 3 (SMT) ==="

for run in 1 2 3 4 5; do
  echo "  Run $run: scenario S1 (single thread on CPU0)"
  run_cpu_burn_with_perf "S1_single" "$run" "$F3_CSV" "$F3_LOG" \
    taskset -c 0 ./build/cpu_burn 5

  echo "  Run $run: scenario S2 (same core: CPU0+CPU1)"
  run_cpu_burn_with_perf "S2_thread0" "$run" "$F3_CSV" "$F3_LOG" \
    taskset -c 0 ./build/cpu_burn 5 &
  pid1=$!
  run_cpu_burn_with_perf "S2_thread1" "$run" "$F3_CSV" "$F3_LOG" \
    taskset -c 1 ./build/cpu_burn 5 &
  pid2=$!
  wait "$pid1" "$pid2"

  echo "  Run $run: scenario S3 (different cores: CPU0+CPU2)"
  run_cpu_burn_with_perf "S3_thread0" "$run" "$F3_CSV" "$F3_LOG" \
    taskset -c 0 ./build/cpu_burn 5 &
  pid3=$!
  run_cpu_burn_with_perf "S3_thread1" "$run" "$F3_CSV" "$F3_LOG" \
    taskset -c 2 ./build/cpu_burn 5 &
  pid4=$!
  wait "$pid3" "$pid4"
done

###############################################################################
# Feature 4: Prefetcher / stride (mem_pattern)
###############################################################################

F4_CSV="results/feature4_prefetch.csv"
F4_LOG="results/feature4_prefetch_perf.log"
echo "pattern,stride,run,runtime_seconds" > "$F4_CSV"
: > "$F4_LOG"

echo "=== Running Feature 4 (prefetch / stride) ==="

# Sequential baseline
for run in 1 2 3; do
  run_with_perf_generic "seq,1" "$run" "$F4_CSV" "$F4_LOG" \
    taskset -c 0 ./build/mem_pattern --size-mb 256 --pattern seq --stride 1 --repeats 5
done

# Strided accesses with increasing stride
for stride in 1 2 4 8 16 32 64 128; do
  for run in 1 2 3; do
    run_with_perf_generic "stride,${stride}" "$run" "$F4_CSV" "$F4_LOG" \
      taskset -c 0 ./build/mem_pattern --size-mb 256 --pattern stride --stride "$stride" --repeats 5
  done
done

# Random access pattern
for run in 1 2 3; do
  run_with_perf_generic "rand,0" "$run" "$F4_CSV" "$F4_LOG" \
    taskset -c 0 ./build/mem_pattern --size-mb 256 --pattern rand --stride 1 --repeats 5
done

echo "All experiments completed. CSV files are in results/."

