# Pin to 1 core, max perf state, isolate noise
sudo cpupower frequency-set -g performance
sudo sysctl -w kernel.nmi_watchdog=0
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space      # optional for repeatability
sudo sh -c 'echo never  > /sys/kernel/mm/transparent_hugepage/defrag'
sudo sh -c 'echo always > /sys/kernel/mm/transparent_hugepage/enabled'  # toggle later for TLB tests

# Run benches with: taskset -c 2 chrt -f 99 <command>         # pin + high prio
