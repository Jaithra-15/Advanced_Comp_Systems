// cpu_burn.cpp
// CPU-bound workload for affinity and SMT experiments.
// Usage: ./cpu_burn [seconds]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <inttypes.h>

int main(int argc, char* argv[]) {
    double seconds = 5.0;
    if (argc >= 2) {
        seconds = std::atof(argv[1]);
        if (seconds <= 0.0) seconds = 5.0;
    }

    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();
    auto t_end   = t_start + std::chrono::duration<double>(seconds);

    volatile double x = 1.0;
    std::uint64_t iters = 0;

    while (clock::now() < t_end) {
        for (int i = 0; i < 1000000; ++i) {
            x = x * 1.0000001 + 0.0000001;
        }
        iters += 1000000;
    }

    auto t_stop = clock::now();
    double elapsed = std::chrono::duration<double>(t_stop - t_start).count();

    std::printf("cpu_burn done: iters=%" PRIu64 ", x=%f\n", iters, x);
    std::printf("RUNTIME_SECONDS %.6f\n", elapsed);
    return 0;
}

