#!/usr/bin/env bash
set -euo pipefail

# Pin later with: taskset -c 0 or numactl --cpunodebind=0 --membind=0
echo "Governor (need sudo to set 'performance'):"
grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true

echo "Transparent Huge Pages:"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true

echo "SMT status:"
lscpu | egrep 'Thread|Core|Socket|NUMA'

echo "Kernel & perf:"
uname -a
perf --version || true
