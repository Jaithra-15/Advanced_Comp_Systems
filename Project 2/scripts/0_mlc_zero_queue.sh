#!/usr/bin/env bash
set -euo pipefail
MLC=${MLC:-./mlc}   # set path
mkdir -p data

# Idle/zero-queue latency (pointer-chasing)
sudo $MLC --idle_latency | tee data/mlc_latency_idle.txt

# Optional: per-socket / matrix if multi-socket
sudo $MLC --latency_matrix | tee data/mlc_latency_matrix.txt
