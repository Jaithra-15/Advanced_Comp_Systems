import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV and force numeric types
df = pd.read_csv("page_miss_results.csv")

# Convert columns to numeric, coerce errors
df['ns_per_access'] = pd.to_numeric(df['ns_per_access'], errors='coerce')
df['pages'] = pd.to_numeric(df['pages'], errors='coerce')

# Drop any rows that could not be converted
df = df.dropna(subset=['ns_per_access', 'pages']).reset_index(drop=True)

# Compute differences between consecutive ns_per_access
df['ns_diff'] = df['ns_per_access'].diff()
df['pages_diff'] = df['pages'].diff()

# Estimate TLB miss impact as slope of latency increase
df['impact_per_page'] = df['ns_diff'] / df['pages_diff']

# Identify working set threshold where latency jumps (TLB exceeded)
threshold_idx = df['impact_per_page'].idxmax()
tlb_threshold_pages = df.loc[threshold_idx, 'pages']
max_impact = df.loc[threshold_idx, 'impact_per_page']

print(f"Estimated TLB threshold (pages): {tlb_threshold_pages}")
print(f"Maximum TLB miss impact (ns per page): {max_impact:.2f}")

# Plot latency vs pages
plt.figure(figsize=(8,5))
plt.plot(df['pages'], df['ns_per_access'], marker='o')
plt.axvline(tlb_threshold_pages, color='r', linestyle='--', label='TLB exceeded')
plt.xlabel("Number of pages accessed")
plt.ylabel("Average ns per access")
plt.title("TLB Miss Impact")
plt.legend()
plt.grid(True)
plt.show()
