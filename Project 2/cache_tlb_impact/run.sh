g++ -O3 -std=c++17 cache_misses.cpp -o cache_misses
./cache_misses 67108864 64 10   # 64MB, stride 64 bytes, 10 repetitions
