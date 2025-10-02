# Install build deps
sudo apt install build-essential flex bison libelf-dev libdw-dev

# Get Microsoft’s WSL2 kernel tree
git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf

# Build perf
make -j$(nproc)

# Install the binary
sudo cp perf /usr/local/bin/
