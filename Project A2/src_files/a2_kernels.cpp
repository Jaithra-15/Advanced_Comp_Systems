#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <omp.h>

#include "a2_kernels.h"
#include "a2_utils.h"

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

static void* aligned_malloc(size_t bytes, size_t alignment) {
  void* p = nullptr;
  if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
  return p;
}

AlignedBuffer make_aligned_f32(size_t count, size_t alignment) {
  AlignedBuffer b;
  b.count = count;
  b.ptr = (float*)aligned_malloc(count * sizeof(float), alignment);
  if (!b.ptr) std::abort();
  return b;
}

void free_aligned(AlignedBuffer& b) {
  std::free(b.ptr);
  b.ptr = nullptr;
  b.count = 0;
}

void fill_random(float* x, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < n; i++) x[i] = dist(rng);
}

void zero_fill(float* x, size_t n) {
  std::memset(x, 0, n * sizeof(float));
}

CSR make_random_csr(int m, int k, double density, const std::string& pattern, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> coldist(0, k - 1);

  CSR A;
  A.m = m; A.k = k;
  A.rowptr.resize((size_t)m + 1, 0);

  std::vector<std::vector<int>> cols((size_t)m);
  std::vector<std::vector<float>> vals((size_t)m);

  if (pattern == "band") {
    int bw = std::max(1, (int)std::round(k * std::min(0.20, std::max(0.01, density * 5.0))));
    for (int i = 0; i < m; i++) {
      int center = (int)((int64_t)i * k / std::max(1, m));
      int lo = std::max(0, center - bw);
      int hi = std::min(k - 1, center + bw);
      int span = hi - lo + 1;
      int target = (int)std::round(span * std::min(1.0, density * (double)k / (double)span));
      target = std::max(1, std::min(span, target));
      cols[(size_t)i].reserve((size_t)target);
      vals[(size_t)i].reserve((size_t)target);
      std::uniform_int_distribution<int> bdist(lo, hi);
      for (int t = 0; t < target; t++) {
        int c = bdist(rng);
        cols[(size_t)i].push_back(c);
        vals[(size_t)i].push_back(vdist(rng));
      }
      std::sort(cols[(size_t)i].begin(), cols[(size_t)i].end());
      cols[(size_t)i].erase(std::unique(cols[(size_t)i].begin(), cols[(size_t)i].end()), cols[(size_t)i].end());
    }
  } else if (pattern == "blockdiag") {
    int blocks = 8;
    int bm = std::max(1, m / blocks);
    int bk = std::max(1, k / blocks);
    for (int i = 0; i < m; i++) {
      int bi = std::min(blocks - 1, i / bm);
      int lo = bi * bk;
      int hi = std::min(k - 1, lo + bk - 1);
      int span = hi - lo + 1;
      int target = (int)std::round(span * std::min(1.0, density * (double)k / (double)span));
      target = std::max(1, std::min(span, target));
      cols[(size_t)i].reserve((size_t)target);
      vals[(size_t)i].reserve((size_t)target);
      std::uniform_int_distribution<int> bdist(lo, hi);
      for (int t = 0; t < target; t++) {
        int c = bdist(rng);
        cols[(size_t)i].push_back(c);
        vals[(size_t)i].push_back(vdist(rng));
      }
      std::sort(cols[(size_t)i].begin(), cols[(size_t)i].end());
      cols[(size_t)i].erase(std::unique(cols[(size_t)i].begin(), cols[(size_t)i].end()), cols[(size_t)i].end());
    }
  } else {
    // uniform
    double expected = (double)m * (double)k * density;
    int nnz = (int)std::round(expected);
    nnz = std::max(m, nnz);
    std::uniform_int_distribution<int> rowdist(0, m - 1);

    for (int t = 0; t < nnz; t++) {
      int r = rowdist(rng);
      int c = coldist(rng);
      cols[(size_t)r].push_back(c);
      vals[(size_t)r].push_back(vdist(rng));
    }

    for (int r = 0; r < m; r++) {
      auto& rc = cols[(size_t)r];
      auto& rv = vals[(size_t)r];
      if (rc.empty()) { rc.push_back(coldist(rng)); rv.push_back(vdist(rng)); }

      std::vector<size_t> idx(rc.size());
      for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return rc[a] < rc[b]; });

      std::vector<int> rc2; rc2.reserve(rc.size());
      std::vector<float> rv2; rv2.reserve(rv.size());
      int last = -1;
      for (size_t t2 = 0; t2 < idx.size(); t2++) {
        int c = rc[idx[t2]];
        if (c == last) continue;
        last = c;
        rc2.push_back(c);
        rv2.push_back(rv[idx[t2]]);
      }
      rc.swap(rc2);
      rv.swap(rv2);
    }
  }

  size_t total = 0;
  for (int i = 0; i < m; i++) {
    A.rowptr[(size_t)i] = (int)total;
    total += cols[(size_t)i].size();
  }
  A.rowptr[(size_t)m] = (int)total;

  A.colidx.resize(total);
  A.values.resize(total);

  size_t p = 0;
  for (int i = 0; i < m; i++) {
    for (size_t t = 0; t < cols[(size_t)i].size(); t++) {
      A.colidx[p] = cols[(size_t)i][t];
      A.values[p] = vals[(size_t)i][t];
      p++;
    }
  }
  return A;
}

size_t csr_nnz(const CSR& A) { return A.values.size(); }

void gemm_tiled_scalar(const float* A, const float* B, float* C,
                       int m, int k, int n, int tileM, int tileK, int tileN) {
  // Parallelize by ii blocks (each thread writes distinct rows of C)
#pragma omp parallel for schedule(static)
  for (int ii = 0; ii < m; ii += tileM) {
    for (int kk = 0; kk < k; kk += tileK) {
      for (int jj = 0; jj < n; jj += tileN) {
        int i_end = std::min(m, ii + tileM);
        int k_end = std::min(k, kk + tileK);
        int j_end = std::min(n, jj + tileN);
        for (int i = ii; i < i_end; i++) {
          for (int t = kk; t < k_end; t++) {
            float a = A[i * k + t];
            const float* b = &B[t * n + jj];
            float* c = &C[i * n + jj];
            for (int j = jj; j < j_end; j++) {
              c[j - jj] += a * b[j - jj];
            }
          }
        }
      }
    }
  }
}

#if defined(__AVX2__)
void gemm_tiled_avx2(const float* A, const float* B, float* C,
                     int m, int k, int n, int tileM, int tileK, int tileN) {
#pragma omp parallel for schedule(static)
  for (int ii = 0; ii < m; ii += tileM) {
    for (int kk = 0; kk < k; kk += tileK) {
      for (int jj = 0; jj < n; jj += tileN) {
        int i_end = std::min(m, ii + tileM);
        int k_end = std::min(k, kk + tileK);
        int j_end = std::min(n, jj + tileN);
        int j_vec_end = jj + ((j_end - jj) / 8) * 8;

        for (int i = ii; i < i_end; i++) {
          for (int t = kk; t < k_end; t++) {
            __m256 a8 = _mm256_set1_ps(A[i * k + t]);
            const float* b = &B[t * n + jj];
            float* c = &C[i * n + jj];

            for (int j = jj; j < j_vec_end; j += 8) {
              __m256 bv = _mm256_loadu_ps(b + (j - jj));
              __m256 cv = _mm256_loadu_ps(c + (j - jj));
              cv = _mm256_fmadd_ps(a8, bv, cv);
              _mm256_storeu_ps(c + (j - jj), cv);
            }
            for (int j = j_vec_end; j < j_end; j++) {
              C[i * n + j] += A[i * k + t] * B[t * n + j];
            }
          }
        }
      }
    }
  }
}
#endif // __AVX2__

void spmm_csr_scalar(const CSR& A, const float* B, float* C, int n,
                     int jblock, LayoutB layoutB) {
  const int m = A.m;
  zero_fill(C, (size_t)m * (size_t)n);

  auto getB = [&](int col, int j)->float {
    if (layoutB == LayoutB::RowMajor) return B[col * n + j];
    return B[j * A.k + col];
  };

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    int p0 = A.rowptr[(size_t)i];
    int p1 = A.rowptr[(size_t)i + 1];
    float* crow = &C[i * n];

    for (int j0 = 0; j0 < n; j0 += jblock) {
      int j1 = std::min(n, j0 + jblock);
      for (int p = p0; p < p1; p++) {
        int col = A.colidx[(size_t)p];
        float a = A.values[(size_t)p];
        for (int j = j0; j < j1; j++) {
          crow[j] += a * getB(col, j);
        }
      }
    }
  }
}

#if defined(__AVX2__)
void spmm_csr_avx2(const CSR& A, const float* B, float* C, int n,
                   int jblock, LayoutB layoutB) {
  if (layoutB != LayoutB::RowMajor) {
    spmm_csr_scalar(A, B, C, n, jblock, layoutB);
    return;
  }

  const int m = A.m;
  zero_fill(C, (size_t)m * (size_t)n);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < m; i++) {
    int p0 = A.rowptr[(size_t)i];
    int p1 = A.rowptr[(size_t)i + 1];
    float* crow = &C[i * n];

    for (int j0 = 0; j0 < n; j0 += jblock) {
      int j1 = std::min(n, j0 + jblock);
      int j_vec_end = j0 + ((j1 - j0) / 8) * 8;

      for (int p = p0; p < p1; p++) {
        int col = A.colidx[(size_t)p];
        float a = A.values[(size_t)p];
        __m256 a8 = _mm256_set1_ps(a);
        const float* brow = &B[col * n];

        for (int j = j0; j < j_vec_end; j += 8) {
          __m256 bv = _mm256_loadu_ps(brow + j);
          __m256 cv = _mm256_loadu_ps(crow + j);
          cv = _mm256_fmadd_ps(a8, bv, cv);
          _mm256_storeu_ps(crow + j, cv);
        }
        for (int j = j_vec_end; j < j1; j++) {
          crow[j] += a * brow[j];
        }
      }
    }
  }
}
#endif // __AVX2__

double stream_triad_bandwidth_gbps(size_t N, int iters, uint64_t seed) {
  (void)seed;
  AlignedBuffer a = make_aligned_f32(N, 64);
  AlignedBuffer b = make_aligned_f32(N, 64);
  AlignedBuffer c = make_aligned_f32(N, 64);

  fill_random(a.ptr, N, 1);
  fill_random(b.ptr, N, 2);
  fill_random(c.ptr, N, 3);

  volatile float s = 1.1f;

  double t0 = now_seconds();
  for (int it = 0; it < iters; it++) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
      a.ptr[i] = b.ptr[i] + s * c.ptr[i];
    }
  }
  double t1 = now_seconds();
  double sec = std::max(1e-9, t1 - t0);

  double bytes_per_iter = (double)N * sizeof(float) * 4.0;
  double gbps = (bytes_per_iter * (double)iters) / sec / 1e9;

  free_aligned(a); free_aligned(b); free_aligned(c);
  return gbps;
}

