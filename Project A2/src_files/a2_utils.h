#pragma once
#include <string>
#include <vector>

double now_seconds();

void percentile_us(const std::vector<double>& samples_sec, double& p50, double& p95, double& p99);

bool file_exists(const std::string& path);

struct PerfCounters {
  bool valid = false;
  double cycles = 0.0;
  double instructions = 0.0;
  double cache_misses = 0.0;
  double llc_load_misses = 0.0;
  double llc_store_misses = 0.0;
  double dtlb_load_misses = 0.0;
};

PerfCounters read_perf_csv(const std::string& perf_csv_path);

