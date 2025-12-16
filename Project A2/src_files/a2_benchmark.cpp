#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "a2_kernels.h"
#include "a2_utils.h"

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def="") {
  for (int i = 1; i + 1 < argc; i++) {
    if (std::string(argv[i]) == key) return std::string(argv[i+1]);
  }
  return def;
}
static int get_arg_i(int argc, char** argv, const std::string& key, int def) {
  std::string s = get_arg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::atoi(s.c_str());
}
static double get_arg_f(int argc, char** argv, const std::string& key, double def) {
  std::string s = get_arg(argc, argv, key, "");
  if (s.empty()) return def;
  return std::atof(s.c_str());
}

static void print_header() {
  std::cout
    << "kernel,variant,layoutB,pattern,m,k,n,density,threads,tileM,tileN,tileK,jblock,seed,run,"
    << "seconds,gflops,nnz,cpnz,ai,bytes_est,bandwidth_GBps,p50_us,p95_us,p99_us,conv_seconds,"
    << "freq_mhz,cycles_est,"
    << "perf_task_clock_ms,perf_context_switches,perf_cpu_migrations,perf_page_faults\n";
}

int main(int argc, char** argv) {
  const std::string kernel = get_arg(argc, argv, "--kernel", "gemm");
  const std::string variant = get_arg(argc, argv, "--variant", "simd");
  const std::string layoutB_s = get_arg(argc, argv, "--layoutB", "row");
  const std::string pattern = get_arg(argc, argv, "--pattern", "uniform");

  const int m = get_arg_i(argc, argv, "--m", 1024);
  const int k = get_arg_i(argc, argv, "--k", 1024);
  const int n = get_arg_i(argc, argv, "--n", 1024);
  const double density = get_arg_f(argc, argv, "--density", 1.0);

  const int threads = get_arg_i(argc, argv, "--threads", 1);
  const int tileM = get_arg_i(argc, argv, "--tileM", 64);
  const int tileN = get_arg_i(argc, argv, "--tileN", 128);
  const int tileK = get_arg_i(argc, argv, "--tileK", 64);
  const int jblock = get_arg_i(argc, argv, "--jblock", 128);

  const uint64_t seed = (uint64_t)get_arg_i(argc, argv, "--seed", 123);
  const int run_id = get_arg_i(argc, argv, "--run", 0);

  const double freq_mhz = get_arg_f(argc, argv, "--freq_mhz", 2400.0);

  // perf fields that are actually measurable in your environment
  const double perf_task_clock_ms = get_arg_f(argc, argv, "--perf_task_clock_ms", 0.0);
  const double perf_context_switches = get_arg_f(argc, argv, "--perf_context_switches", 0.0);
  const double perf_cpu_migrations = get_arg_f(argc, argv, "--perf_cpu_migrations", 0.0);
  const double perf_page_faults = get_arg_f(argc, argv, "--perf_page_faults", 0.0);

  const int header = get_arg_i(argc, argv, "--header", 0);
  if (header) {
    print_header();
    return 0;
  }

  omp_set_num_threads(std::max(1, threads));

  LayoutB layoutB = (layoutB_s == "col") ? LayoutB::ColMajor : LayoutB::RowMajor;

  double seconds = 0.0;
  double gflops = 0.0;
  size_t nnz = 0;
  double conv_seconds = 0.0;

  double bytes_est = 0.0;
  double ai = 0.0;
  double bw_gbps = 0.0;

  std::vector<double> call_times;
  call_times.reserve(32);

  if (kernel == "stream") {
    double gbps = stream_triad_bandwidth_gbps((size_t)64 * 1024 * 1024 / sizeof(float), 10, seed);
    seconds = 0.0;
    gflops = 0.0;
    nnz = 0;
    bytes_est = 0.0;
    ai = 0.0;
    bw_gbps = gbps;
    call_times = {0.0};
  } else if (kernel == "gemm") {
    AlignedBuffer A = make_aligned_f32((size_t)m * (size_t)k, 64);
    AlignedBuffer B = make_aligned_f32((size_t)k * (size_t)n, 64);
    AlignedBuffer C = make_aligned_f32((size_t)m * (size_t)n, 64);

    fill_random(A.ptr, A.count, seed ^ 0xA5A5u);
    fill_random(B.ptr, B.count, seed ^ 0x5A5Au);

    const int reps = 15;
    for (int r = 0; r < reps; r++) {
      zero_fill(C.ptr, C.count);
      double t0 = now_seconds();
#if defined(__AVX2__)
      if (variant == "simd") gemm_tiled_avx2(A.ptr, B.ptr, C.ptr, m, k, n, tileM, tileK, tileN);
      else                   gemm_tiled_scalar(A.ptr, B.ptr, C.ptr, m, k, n, tileM, tileK, tileN);
#else
      gemm_tiled_scalar(A.ptr, B.ptr, C.ptr, m, k, n, tileM, tileK, tileN);
#endif
      double t1 = now_seconds();
      call_times.push_back(t1 - t0);
    }

    std::vector<double> tmp = call_times;
    std::sort(tmp.begin(), tmp.end());
    seconds = tmp[tmp.size() / 2];

    const double flops = 2.0 * (double)m * (double)k * (double)n;
    gflops = flops / std::max(1e-12, seconds) / 1e9;

    bytes_est = 4.0 * ((double)m * k + (double)k * n + 2.0 * (double)m * n);
    ai = flops / std::max(1.0, bytes_est);
    bw_gbps = (bytes_est / std::max(1e-12, seconds)) / 1e9;

    free_aligned(A); free_aligned(B); free_aligned(C);
  } else if (kernel == "spmm_csr") {
    double t0c = now_seconds();
    CSR A = make_random_csr(m, k, density, pattern, seed);
    double t1c = now_seconds();
    conv_seconds = t1c - t0c;

    nnz = csr_nnz(A);

    AlignedBuffer B = make_aligned_f32((size_t)k * (size_t)n, 64);
    AlignedBuffer C = make_aligned_f32((size_t)m * (size_t)n, 64);
    fill_random(B.ptr, B.count, seed ^ 0x1234u);

    const int reps = 20;
    for (int r = 0; r < reps; r++) {
      double t0 = now_seconds();
#if defined(__AVX2__)
      if (variant == "simd") spmm_csr_avx2(A, B.ptr, C.ptr, n, jblock, layoutB);
      else                   spmm_csr_scalar(A, B.ptr, C.ptr, n, jblock, layoutB);
#else
      spmm_csr_scalar(A, B.ptr, C.ptr, n, jblock, layoutB);
#endif
      double t1 = now_seconds();
      call_times.push_back(t1 - t0);
    }

    std::vector<double> tmp = call_times;
    std::sort(tmp.begin(), tmp.end());
    seconds = tmp[tmp.size() / 2];

    const double flops = 2.0 * (double)nnz * (double)n;
    gflops = flops / std::max(1e-12, seconds) / 1e9;

    bytes_est = (double)nnz * 8.0 + (double)nnz * (double)n * 4.0 + (double)m * (double)n * 8.0;
    ai = flops / std::max(1.0, bytes_est);
    bw_gbps = (bytes_est / std::max(1e-12, seconds)) / 1e9;

    free_aligned(B); free_aligned(C);
  } else {
    std::cerr << "Unknown --kernel\n";
    return 2;
  }

  double p50 = 0.0, p95 = 0.0, p99 = 0.0;
  percentile_us(call_times, p50, p95, p99);  // from a2_utils.*

  const double cycles_est = seconds * (freq_mhz * 1e6);
  const double cpnz = (nnz > 0) ? (cycles_est / (double)nnz) : 0.0;

  std::cout
    << kernel << "," << variant << "," << layoutB_s << "," << pattern << ","
    << m << "," << k << "," << n << ","
    << density << ","
    << threads << ","
    << tileM << "," << tileN << "," << tileK << "," << jblock << ","
    << seed << "," << run_id << ","
    << seconds << "," << gflops << ","
    << nnz << "," << cpnz << ","
    << ai << "," << bytes_est << "," << bw_gbps << ","
    << p50 << "," << p95 << "," << p99 << ","
    << conv_seconds << ","
    << freq_mhz << "," << cycles_est << ","
    << perf_task_clock_ms << "," << perf_context_switches << ","
    << perf_cpu_migrations << "," << perf_page_faults
    << "\n";

  return 0;
}

