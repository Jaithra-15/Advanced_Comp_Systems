#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cmath>

constexpr size_t PAGE_SIZE = 4096; // 4 KB
constexpr size_t STRIDE = PAGE_SIZE / sizeof(int);

int main() {
    size_t max_pages = 1 << 16; // up to 64K pages
    std::vector<int> arr(max_pages * STRIDE, 0);

    std::cout << "Pages\tTime_per_access(ns)\n";

    for (size_t pages = 16; pages <= max_pages; pages *= 2) {
        volatile int sink = 0;

        auto start = std::chrono::high_resolution_clock::now();
        // pointer-chasing loop
        for (size_t repeat = 0; repeat < 100; repeat++) {
            for (size_t i = 0; i < pages * STRIDE; i += STRIDE) {
                sink += arr[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double ns_per_access = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
                               / double(pages * 100);
        std::cout << pages << "\t" << ns_per_access << "\n";
    }

    return 0;
}
