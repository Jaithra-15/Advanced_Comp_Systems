// cache_misses.cpp
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iostream>

static inline uint64_t nsec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

int main(int argc, char** argv) {
    size_t bytes  = (argc > 1) ? std::stoull(argv[1]) : (64ull << 20); // default 64 MB
    size_t stride = (argc > 2) ? std::stoull(argv[2]) : 64;            // stride in bytes
    size_t iters  = (argc > 3) ? std::stoull(argv[3]) : 10;            // repetitions

    size_t n = bytes / sizeof(uint64_t);

    // allocate 4096-aligned memory
    uint64_t* a = static_cast<uint64_t*>(
        aligned_alloc(4096, n * sizeof(uint64_t))
    );
    if (!a) {
        std::perror("aligned_alloc failed");
        return 1;
    }

    for (size_t i = 0; i < n; i++) a[i] = i;

    volatile uint64_t sink = 0;
    for (size_t i = 0; i < n; i += 64 / 8) sink += a[i]; // touch each page

    uint64_t best = ~0ull;
    size_t step = stride / sizeof(uint64_t);

    for (size_t r = 0; r < iters; r++) {
        uint64_t t0 = nsec();
        for (size_t i = 0; i < n; i += step) sink += a[i];
        uint64_t t1 = nsec();
        if (t1 - t0 < best) best = t1 - t0;
    }

    double accesses = static_cast<double>(n) / step;
    double ns_per_access = static_cast<double>(best) / accesses;

    std::cout << "bytes=" << bytes
              << ", stride=" << stride
              << ", ns_per_access=" << ns_per_access << "\n";

    std::cerr << "sink=" << sink << "\n";

    free(a);
    return 0;
}
