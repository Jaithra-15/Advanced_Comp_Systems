#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# === USER PARAMETERS (edit these for your machine) ===
MEM_BANDWIDTH = 60.0   # GB/s from Project #2 (example)
CPU_PEAK_GFLOPS = 800  # peak FLOPs (example: AVX2/AVX-512 calculation)

# Arithmetic intensity for SAXPY (f32)
AI = 2.0 / 12.0  # FLOPs per byte
print(f"Arithmetic Intensity (AI) = {AI:.3f} FLOPs/byte")

# === Load CSV ===
df = pd.read_csv("full_sweep_simd.csv")

# === Filter for saxpy, f32, misaligned = false, tail_multiple = true ===
mask = (
    (df["kernel"] == "saxpy") &
    (df["dtype"] == "f32") &
    (df["misaligned"] == False) &
    (df["tail_multiple"] == True)
)
df = df[mask]

# === Aggregate mean GFLOPs per N ===
grouped = df.groupby("N", as_index=False)["GFLOPs"].mean()

# === Compute memory roof ===
roof_mem = AI * MEM_BANDWIDTH  # GFLOPs/s
print(f"Memory roof = {roof_mem:.2f} GFLOP/s")
print(f"Compute roof = {CPU_PEAK_GFLOPS:.2f} GFLOP/s")

# === Plot Roofline ===
plt.figure(figsize=(8, 6))

# Plot rooflines
plt.axhline(CPU_PEAK_GFLOPS, color="red", linestyle="--", label="Compute Roof")
plt.axhline(roof_mem, color="blue", linestyle="--", label="Memory Roof")

# Plot measured performance (all points at same AI)
plt.scatter([AI] * len(grouped), grouped["GFLOPs"], marker="o", color="black", label="Measured (SAXPY f32)")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
plt.ylabel("Performance (GFLOP/s)")
plt.title("Roofline Model for SAXPY (f32)")
plt.legend()
plt.grid(True, which="both", linestyle=":")

plt.tight_layout()
plt.savefig("roofline_saxpy.png", dpi=300)
plt.show()
