for b in 64K 256K 1M 4M 16M 64M 256M 1G; do
    ./cache_misses $(numfmt --from=iec $b) 64 5
done
