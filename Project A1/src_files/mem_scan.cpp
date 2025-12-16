// mem_scan.cpp
// Large memory scan for THP experiments.
// Usage example:
//   ./mem_scan --size-mb 2048 --pattern seq   --stride 1   --repeats 3
//   ./mem_scan --size-mb 2048 --pattern rand              --repeats 3
//   ./mem_scan --size-mb 2048 --pattern stride --stride 64 --repeats 3

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

static void shuffle_indices(std::size_t* idx, std::size_t n) {
    for (std::size_t i = n - 1; i > 0; --i) {
        std::size_t j = static_cast<std::size_t>(std::rand()) % (i + 1);
        std::size_t tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

int main(int argc, char* argv[]) {
    std::size_t size_mb = 2048;   // 2 GB by default
    const char* pattern = "seq";  // seq | stride | rand
    std::size_t stride = 1;
    int repeats = 3;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--size-mb") == 0 && i + 1 < argc) {
            size_mb = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--pattern") == 0 && i + 1 < argc) {
            pattern = argv[++i];
        } else if (std::strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            stride = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        }
    }

    std::size_t bytes = size_mb * 1024ULL * 1024ULL;
    std::size_t n = bytes / sizeof(std::uint64_t);

    std::uint64_t* a =
        static_cast<std::uint64_t*>(std::malloc(n * sizeof(std::uint64_t)));
    if (!a) {
        std::fprintf(stderr, "Allocation failed for %zu bytes\n", bytes);
        return 1;
    }

    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::uint64_t>(i);
    }

    std::size_t* indices = nullptr;
    if (std::strcmp(pattern, "rand") == 0) {
        indices = static_cast<std::size_t*>(
            std::malloc(n * sizeof(std::size_t)));
        if (!indices) {
            std::fprintf(stderr, "Index allocation failed\n");
            std::free(a);
            return 1;
        }
        for (std::size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        shuffle_indices(indices, n);
    }

    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();

    volatile std::uint64_t sum = 0;
    for (int r = 0; r < repeats; ++r) {
        if (std::strcmp(pattern, "seq") == 0) {
            for (std::size_t i = 0; i < n; ++i) {
                sum += a[i];
            }
        } else if (std::strcmp(pattern, "stride") == 0) {
            for (std::size_t i = 0; i < n; i += stride) {
                sum += a[i];
            }
        } else if (std::strcmp(pattern, "rand") == 0) {
            for (std::size_t i = 0; i < n; ++i) {
                sum += a[indices[i]];
            }
        } else {
            std::fprintf(stderr, "Unknown pattern '%s'\n", pattern);
            break;
        }
    }

    auto t_stop = clock::now();
    double elapsed =
        std::chrono::duration<double>(t_stop - t_start).count();

    std::printf("mem_scan done: size_mb=%zu, pattern=%s, stride=%zu, "
                "repeats=%d, sum=%" PRIu64 "\n",
                size_mb, pattern, stride, repeats, sum);
    std::printf("RUNTIME_SECONDS %.6f\n", elapsed);

    std::free(indices);
    std::free(a);
    return 0;
}

