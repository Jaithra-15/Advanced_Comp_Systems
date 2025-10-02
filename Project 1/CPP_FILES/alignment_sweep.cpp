#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;
using namespace std::chrono;

// -------------------------------
// Build modes (choose via g++)
//   Scalar:   -O3 -fno-tree-vectorize -march=native -std=c++17
//   SIMD:     -O3 -fopenmp-simd        -march=native -std=c++17
// Optional:   -ffast-math (changes FLOP counts!)
// Vector reports:
//   GCC:   -fopt-info-vec-optimized -fopt-info-vec-missed
// -------------------------------

// Aligned alloc
template<typename T>
T* xaligned_alloc(size_t n, size_t alignment = 64) {
    void* p = nullptr;
    if (posix_memalign(&p, alignment, n * sizeof(T)) != 0) throw bad_alloc();
    return reinterpret_cast<T*>(p);
}

template<typename T>
inline T* assume_aligned(T* p, size_t alignment = 64) {
#if defined(__GNUC__) || defined(__clang__)
    return reinterpret_cast<T*>(__builtin_assume_aligned(p, alignment));
#else
    return p;
#endif
}

// -------------------------------
// Experimental knobs
// -------------------------------
enum class DType { f32, f64, i32 };
enum class Access { Unit, Strided, Gather };
struct Config {
    DType dtype = DType::f32;
    Access access = Access::Unit;
    size_t N = 1 << 18;
    size_t stride = 1;             // for Access::Strided
    bool misaligned = false;       // misalign base by 1 element
    bool tail_multiple = true;     // whether N is multiple of vector width
    int trials = 7;             
    double cpu_ghz = 2.6;         // changes based on the pinned clock speed
    string kernel = "saxpy";       // "saxpy", "stencil", "elemmul"
};

// arithmetic intensity helper (FLOPs / bytes)
struct Metrics {
    double time_s = 0.0;
    double gflops = 0.0;
    double gibs   = 0.0;
    double ns_per_elem = 0.0;
    double cpe = -1.0; // cycles per element (if cpu_ghz>0)
    double arith_intensity = 0.0;
    double checksum = 0.0;
};

// FLOP counts per element (approx; with -ffast-math stencil uses mul not div)
template<typename T> inline int flops_saxpy() { return 2; }          // a*x + y
template<typename T> inline int flops_elem_mul() { return 1; }       // x*y
template<typename T> inline int flops_stencil() { return std::is_floating_point<T>::value ? 3 : 0; } // (a+b+c)*(1/3)

// Bytes touched per element (conservative stream estimate)
struct BytesModel { double bytes; const char* note; };
template<typename T> inline BytesModel bytes_saxpy()   { return { 3.0 * sizeof(T), "R x, R/W y (RMW)" }; }
template<typename T> inline BytesModel bytes_elemmul() { return { 3.0 * sizeof(T), "R x, R y, W z" }; }
template<typename T> inline BytesModel bytes_stencil() { return { 2.0 * sizeof(T), "R in (amortized), W out" }; } // amortized between neighbors

// Index builder for gather-like access
template<typename IndexT>
vector<IndexT> make_gather_indices(size_t N, size_t stride) {
    vector<IndexT> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = (i * stride) % N;
    return idx;
}

// -------------------------------
// Kernels (templated by type + access)
// Each kernel variant must operate on x,y,z (or x,out) views dictated by access pattern.
// -------------------------------
template<typename T>
inline void kernel_saxpy(const T a, const T* __restrict x, T* __restrict y,
                         const size_t N, const Access access, const size_t stride,
                         const uint32_t* __restrict gidx32, const uint64_t* __restrict gidx64)
{
    auto aligned64 = [](const void* p) -> bool {
        return (reinterpret_cast<uintptr_t>(p) & (64 - 1)) == 0;
    };

    switch (access) {
        case Access::Unit: {
            if (aligned64(x) && aligned64(y)) {
                // use aligned pragma only when truly aligned
                const T* __restrict xa = assume_aligned(const_cast<T*>(x));
                T* __restrict ya = assume_aligned(y);
                #pragma omp simd aligned(xa, ya:64)
                for (size_t i = 0; i < N; ++i) ya[i] = a * xa[i] + ya[i];
            } else {
                // safe unaligned path
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) y[i] = a * x[i] + y[i];
            }
        } break;

        case Access::Strided: {
            if (stride == 1) {
                // same as Unit case
                if (aligned64(x) && aligned64(y)) {
                    const T* __restrict xa = assume_aligned(const_cast<T*>(x));
                    T* __restrict ya = assume_aligned(y);
                    #pragma omp simd aligned(xa, ya:64)
                    for (size_t i = 0; i < N; ++i) ya[i] = a * xa[i] + ya[i];
                } else {
                    #pragma omp simd
                    for (size_t i = 0; i < N; ++i) y[i] = a * x[i] + y[i];
                }
            } else {
                // bounds-safe strided traversal (no i*stride overflow)
                #pragma omp simd
                for (size_t j = 0; j < N; j += stride) y[j] = a * x[j] + y[j];
            }
        } break;

        case Access::Gather: {
            if constexpr (sizeof(size_t) == 8) {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx64[i];
                    if (j < N) y[j] = a * x[j] + y[j];
                }
            } else {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx32[i];
                    if (j < N) y[j] = a * x[j] + y[j];
                }
            }
        } break;
    }
}


template<typename T>
inline void kernel_stencil(const T* __restrict in, T* __restrict out,
                           const size_t N, const Access access, const size_t stride,
                           const uint32_t* __restrict gidx32, const uint64_t* __restrict gidx64)
{
    auto aligned64 = [](const void* p) -> bool {
        return (reinterpret_cast<uintptr_t>(p) & (64 - 1)) == 0;
    };

    // define edges
    if (N) { out[0] = 0; if (N > 1) out[N-1] = 0; }
    if (N < 3) return;

    switch (access) {
        case Access::Unit: {
            if (aligned64(in) && aligned64(out)) {
                const T* __restrict ia = assume_aligned(const_cast<T*>(in));
                T* __restrict oa = assume_aligned(out);
                #pragma omp simd aligned(ia, oa:64)
                for (size_t i = 1; i < N - 1; ++i)
                    oa[i] = (ia[i-1] + ia[i] + ia[i+1]) * T(1.0/3.0);
            } else {
                #pragma omp simd
                for (size_t i = 1; i < N - 1; ++i)
                    out[i] = (in[i-1] + in[i] + in[i+1]) * T(1.0/3.0);
            }
        } break;

        case Access::Strided: {
            #pragma omp simd
            for (size_t j = stride; j < N - 1; j += stride)
                out[j] = (in[j-1] + in[j] + in[j+1]) * T(1.0/3.0);
        } break;

        case Access::Gather: {
            if constexpr (sizeof(size_t) == 8) {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx64[i];
                    if (j == 0 || j+1 >= N) continue;
                    out[j] = (in[j-1] + in[j] + in[j+1]) * T(1.0/3.0);
                }
            } else {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx32[i];
                    if (j == 0 || j+1 >= N) continue;
                    out[j] = (in[j-1] + in[j] + in[j+1]) * T(1.0/3.0);
                }
            }
        } break;
    }
}


template<typename T>
inline void kernel_elemmul(const T* __restrict x, const T* __restrict y, T* __restrict z,
                           const size_t N, const Access access, const size_t stride,
                           const uint32_t* __restrict gidx32, const uint64_t* __restrict gidx64)
{
    auto aligned64 = [](const void* p) -> bool {
        return (reinterpret_cast<uintptr_t>(p) & (64 - 1)) == 0;
    };

    switch (access) {
        case Access::Unit: {
            if (aligned64(x) && aligned64(y) && aligned64(z)) {
                const T* __restrict xa = assume_aligned(const_cast<T*>(x));
                const T* __restrict ya = assume_aligned(const_cast<T*>(y));
                T* __restrict za = assume_aligned(z);
                #pragma omp simd aligned(xa, ya, za:64)
                for (size_t i = 0; i < N; ++i) za[i] = xa[i] * ya[i];
            } else {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) z[i] = x[i] * y[i];
            }
        } break;

        case Access::Strided: {
            #pragma omp simd
            for (size_t j = 0; j < N; j += stride)
                z[j] = x[j] * y[j];
        } break;

        case Access::Gather: {
            if constexpr (sizeof(size_t) == 8) {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx64[i];
                    if (j < N) z[j] = x[j] * y[j];
                }
            } else {
                #pragma omp simd
                for (size_t i = 0; i < N; ++i) {
                    size_t j = gidx32[i];
                    if (j < N) z[j] = x[j] * y[j];
                }
            }
        } break;
    }
}


// checksum to defeat DCE
template<typename T>
double checksum(const T* p, size_t n) {
    double s = 0.0;
    #pragma omp simd reduction(+:s)
    for (size_t i = 0; i < n; ++i) s += double(p[i]);
    return s;
}

// parse helpers
static inline bool starts_with(const string& s, const char* pfx) {
    return s.rfind(pfx, 0) == 0;
}

Config parse_cli(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        string arg(argv[i]);
        if (starts_with(arg, "--kernel="))   c.kernel = arg.substr(9);
        else if (starts_with(arg, "--dtype=")) {
            auto v = arg.substr(8);
            if (v=="f32") c.dtype=DType::f32; else if (v=="f64") c.dtype=DType::f64; else if (v=="i32") c.dtype=DType::i32;
        } else if (starts_with(arg, "--access=")) {
            auto v = arg.substr(9);
            if (v=="unit") c.access=Access::Unit; else if (v=="strided") c.access=Access::Strided; else if (v=="gather") c.access=Access::Gather;
        } else if (starts_with(arg, "--stride=")) c.stride = stoul(arg.substr(9));
        else if (starts_with(arg, "--N=")) c.N = stoull(arg.substr(4));
        else if (starts_with(arg, "--misaligned=")) c.misaligned = (arg.substr(13)=="1" || arg.substr(13)=="true");
        else if (starts_with(arg, "--tail_multiple=")) c.tail_multiple = (arg.substr(16)=="1" || arg.substr(16)=="true");
        else if (starts_with(arg, "--trials=")) c.trials = stoi(arg.substr(9));
        else if (starts_with(arg, "--cpu_ghz=")) c.cpu_ghz = stod(arg.substr(10));
    }
    return c;
}

// adjust N to create a tail (non-multiple of common vector widths)
size_t adjust_N_for_tail(size_t N, bool tail_multiple, size_t elem_bytes) {
    if (tail_multiple) return N;
    // create an awkward remainder vs 32B/64B vectors
    // e.g., add +3 elements so it's not divisible by 8 or 16 lanes
    return N + 3;
}

template<typename T>
vector<Metrics> run_one(const Config& cfg) {
    using I32 = uint32_t; 
    using I64 = uint64_t;

    // Adjust N for tail handling
    size_t N = adjust_N_for_tail(cfg.N, cfg.tail_multiple, sizeof(T));

    // Determine maximum SIMD width in elements (assume AVX-512 max 64 bytes)
    constexpr size_t MAX_SIMD_BYTES = 64;
    const size_t MAX_SIMD_ELEMS = MAX_SIMD_BYTES / sizeof(T);

    // Extra padding: 256 + max SIMD elements to safely handle tails & misalignment
    size_t mis = cfg.misaligned ? 1 : 0;
    const size_t PAD = 256 + MAX_SIMD_ELEMS + mis;

    // Allocate aligned buffers
    T* x = xaligned_alloc<T>(N + PAD);
    T* y = xaligned_alloc<T>(N + PAD);
    T* z = xaligned_alloc<T>(N + PAD);

    // Apply misalignment
    T* X = x + mis;
    T* Y = y + mis;
    T* Z = z + mis;


    // Initialize full buffer including padding
    for (size_t i = 0; i < N + PAD; ++i) {
        if constexpr (std::is_floating_point<T>::value) {
            x[i] = T(std::sin(0.001 * i));
            y[i] = T(std::cos(0.001 * i));
            z[i] = T(0);
        } else {
            x[i] = T((i * 1315423911u) & 0xFFFF);
            y[i] = T(((i+13) * 2654435761u) & 0xFFFF);
            z[i] = T(0);
        }
    }

    // Gather indices if needed
    vector<I32> g32; 
    vector<I64> g64;
    const I32* g32p = nullptr; 
    const I64* g64p = nullptr;
    if (cfg.access == Access::Gather) {
        if (sizeof(size_t) == 8) { 
            g64 = make_gather_indices<I64>(N, std::max<size_t>(1, cfg.stride)); 
            g64p = g64.data(); 
        } else { 
            g32 = make_gather_indices<I32>(N, std::max<size_t>(1, cfg.stride)); 
            g32p = g32.data(); 
        }
    }



    // Kernel FLOP/byte model
    int flops_per_elem = 0;
    BytesModel bmod{0, ""};
    if (cfg.kernel == "saxpy")        { flops_per_elem = flops_saxpy<T>();     bmod = bytes_saxpy<T>(); }
    else if (cfg.kernel == "elemmul") { flops_per_elem = flops_elem_mul<T>();  bmod = bytes_elemmul<T>(); }
    else                               { flops_per_elem = flops_stencil<T>();   bmod = bytes_stencil<T>(); }

    // Warmup run
    if (cfg.kernel == "saxpy")        kernel_saxpy<T>(T(2), X, Y, N, cfg.access, cfg.stride, g32p, g64p);
    else if (cfg.kernel == "elemmul") kernel_elemmul<T>(X, Y, Z, N, cfg.access, cfg.stride, g32p, g64p);
    else                               kernel_stencil<T>(X, Z, N, cfg.access, cfg.stride, g32p, g64p);
    // Collect trial results
    vector<Metrics> results;
    results.reserve(cfg.trials);


    for (int t = 0; t < cfg.trials; ++t) {
        auto t0 = std::chrono::steady_clock::now();

        if (cfg.kernel == "saxpy")        kernel_saxpy<T>(T(2), X, Y, N, cfg.access, cfg.stride, g32p, g64p);
        else if (cfg.kernel == "elemmul") kernel_elemmul<T>(X, Y, Z, N, cfg.access, cfg.stride, g32p, g64p);
        else                               kernel_stencil<T>(X, Z, N, cfg.access, cfg.stride, g32p, g64p);

        auto t1 = std::chrono::steady_clock::now();

        Metrics m{};
        m.time_s = std::chrono::duration<double>(t1 - t0).count();
        const T* OUT = (cfg.kernel=="saxpy") ? Y : Z;
        m.checksum = checksum(OUT, N);
        m.gflops = (double(flops_per_elem) * double(N)) / (m.time_s * 1e9);
        double bytes = bmod.bytes * double(N);
        m.gibs = (bytes / (1<<30)) / m.time_s;
        m.ns_per_elem = (m.time_s * 1e9) / double(N);
        if (cfg.cpu_ghz > 0.0) m.cpe = (m.ns_per_elem * cfg.cpu_ghz);
        m.arith_intensity = (flops_per_elem) / bmod.bytes;

        results.push_back(m);
    }

    free(x); free(y); free(z);
    return results;
}



int main(int argc, char** argv) {


    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Config cfg = parse_cli(argc, argv);

    // Kernel comes from CLI
    string k = cfg.kernel;

    // Logarithmic sweep of N
    size_t Nmin = 1000;
    size_t Nmax = 15000000;
    int steps = 30; // how many distinct N values

    vector<size_t> Ns;
    double log_min = log10((double)Nmin);
    double log_max = log10((double)Nmax);
    for (int i = 0; i < steps; i++) {
        double logN = log_min + (log_max - log_min) * i / (steps - 1);
        size_t N = (size_t)round(pow(10.0, logN));
        if (Ns.empty() || N != Ns.back()) Ns.push_back(N);
    }

    ofstream ofs("alignment_sweep.csv");
    if (!ofs.is_open()) {
        cerr << "Error: could not open alignment_sweep.csv\n";
        return 1;
    }



    ofs.setf(std::ios::fixed);
    ofs.precision(6);
    ofs << "kernel,dtype,N,trial,misaligned,tail_multiple,GFLOPs,CPE\n";

    for (int mis = 0; mis <=1; ++mis) {
        for (int tail = 0; tail <= 1; ++tail) {
            cfg.misaligned = mis;
            cfg.tail_multiple = tail;
            for (auto N : Ns) {
                cfg.N = N;
                switch (cfg.dtype) {
                    case DType::f32: {
                        auto runs = run_one<float>(cfg);
                        for (int t = 0; t < (int)runs.size(); ++t) {

                            ofs << k << ",f32," << N << "," << t << ","
                                << mis << "," << tail << ","
                                << runs[t].gflops << "," << runs[t].cpe << "\n";
                        }
                    } break;
                    case DType::f64: {
                        auto runs = run_one<double>(cfg);
                        for (int t = 0; t < (int)runs.size(); ++t) {
                            ofs << k << ",f64," << N << "," << t << ","
                                << mis << "," << tail << ","
                                << runs[t].gflops << "," << runs[t].cpe << "\n";
                        }
                    } break;
                    case DType::i32: {
                        auto runs = run_one<int32_t>(cfg);
                        for (int t = 0; t < (int)runs.size(); ++t) {
                            ofs << k << ",i32," << N << "," << t << ","
                                << mis << "," << tail << ","
                                << runs[t].gflops << "," << runs[t].cpe << "\n";
                        }
                    } break;
                }
            }
        }
    }

    ofs.close();
    cerr << "Wrote alignment_sweep.csv with 4 variants (aligned/misaligned × tail/no-tail)\n";
    return 0;
}
