#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "a2_utils.h"

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void percentile_us(const std::vector<double>& samples_sec, double& p50, double& p95, double& p99) {
  if (samples_sec.empty()) { p50 = p95 = p99 = 0.0; return; }
  std::vector<double> v = samples_sec;
  std::sort(v.begin(), v.end());
  auto pick = [&](double q)->double{
    double idx = q * (v.size() - 1);
    size_t i = (size_t)idx;
    double frac = idx - (double)i;
    double a = v[i];
    double b = v[std::min(i + 1, v.size() - 1)];
    return (a + frac * (b - a)) * 1e6; // sec -> us
  };
  p50 = pick(0.50);
  p95 = pick(0.95);
  p99 = pick(0.99);
}

bool file_exists(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fclose(f);
  return true;
}

static bool parse_perf_csv_value(const char* perf_csv_path, const char* event, double& out_val) {
  // perf stat -x, line format is typically:
  // value,unit,event, ...
  FILE* f = std::fopen(perf_csv_path, "r");
  if (!f) return false;

  char line[4096];
  bool ok = false;

  while (std::fgets(line, sizeof(line), f)) {
    if (!std::strstr(line, event)) continue;

    // tokenize by commas (C function, global namespace)
    char* save = nullptr;
    char* tok = ::strtok_r(line, ",", &save);
    if (!tok) continue;

    // skip "<not supported>" / "<not counted>"
    if (std::strchr(tok, '<')) continue;

    out_val = std::atof(tok);
    ok = true;
    break;
  }

  std::fclose(f);
  return ok;
}

PerfCounters read_perf_csv(const std::string& perf_csv_path) {
  PerfCounters pc{};
  pc.valid = false;

  double v = 0.0;
  bool any = false;

  if (parse_perf_csv_value(perf_csv_path.c_str(), "cycles", v)) { pc.cycles = v; any = true; }
  if (parse_perf_csv_value(perf_csv_path.c_str(), "instructions", v)) { pc.instructions = v; any = true; }
  if (parse_perf_csv_value(perf_csv_path.c_str(), "cache-misses", v)) { pc.cache_misses = v; any = true; }
  if (parse_perf_csv_value(perf_csv_path.c_str(), "LLC-load-misses", v)) { pc.llc_load_misses = v; any = true; }
  if (parse_perf_csv_value(perf_csv_path.c_str(), "LLC-store-misses", v)) { pc.llc_store_misses = v; any = true; }
  if (parse_perf_csv_value(perf_csv_path.c_str(), "dTLB-load-misses", v)) { pc.dtlb_load_misses = v; any = true; }

  pc.valid = any;
  return pc;
}

