#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct AlignedBuffer {
  float* ptr = nullptr;
  size_t count = 0;
};

AlignedBuffer make_aligned_f32(size_t count, size_t alignment);
void free_aligned(AlignedBuffer& b);

void fill_random(float* x, size_t n, uint64_t seed);
void zero_fill(float* x, size_t n);

struct CSR {
  int m = 0;
  int k = 0;
  std::vector<int> rowptr;
  std::vector<int> colidx;
  std::vector<float> values;
};

CSR make_random_csr(int m, int k, double density, const std::string& pattern, uint64_t seed);
size_t csr_nnz(const CSR& A);

enum class LayoutB { RowMajor, ColMajor };

void gemm_tiled_scalar(const float* A, const float* B, float* C, int m, int k, int n,
                       int tileM, int tileK, int tileN);

#if defined(__AVX2__)
void gemm_tiled_avx2(const float* A, const float* B, float* C, int m, int k, int n,
                     int tileM, int tileK, int tileN);

void spmm_csr_avx2(const CSR& A, const float* B, float* C, int n,
                   int jblock, LayoutB layoutB);
#endif

void spmm_csr_scalar(const CSR& A, const float* B, float* C, int n,
                     int jblock, LayoutB layoutB);

double stream_triad_bandwidth_gbps(size_t N, int iters, uint64_t seed);

