import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("data/kernel_perf.csv")
# Compute miss rate if your CSV includes both refs & misses; else precompute.
# Plot GFLOP/s vs cache-miss rate; or DTLB MPKI vs GFLOP/s
