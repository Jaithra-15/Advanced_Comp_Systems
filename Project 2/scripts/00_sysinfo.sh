#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
{
  date
  uname -a
  lscpu
  numactl -H || true
} > data/system_info.txt
