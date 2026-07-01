#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color
RED='\033[0;31m'
YELLOW='\033[1;33m'

echo -e "${GREEN}=== HELIX Benchmark Runner ===${NC}"

# Path to workspace root
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Determine build directory
BUILD_DIR=""
if [ -d "out/build/HELIX" ]; then
    BUILD_DIR="out/build/HELIX"
elif [ -d "build" ]; then
    BUILD_DIR="build"
else
    echo -e "${YELLOW}Build directory not found. Running build.sh first...${NC}"
    if [ -f "./build.sh" ]; then
        # Ensure build.sh is executable
        chmod +x ./build.sh
        ./build.sh
        if [ -d "out/build/HELIX" ]; then
            BUILD_DIR="out/build/HELIX"
        else
            BUILD_DIR="build"
        fi
    else
        echo -e "${RED}Error: Build directory not found and build.sh is missing. Please build the project first.${NC}" >&2
        exit 1
    fi
fi

# Ensure output directory exists
mkdir -p "$WORKSPACE_DIR/output"

# Array of benchmark executables
BENCHMARKS=(
    "matmul_benchmark"
    "tensor_ops_benchmark"
    "nn_benchmark"
    "training_benchmark"
)

# Run each benchmark
for BENCHMARK in "${BENCHMARKS[@]}"; do
    BENCHMARK_PATH="$WORKSPACE_DIR/$BUILD_DIR/benchmark/$BENCHMARK"
    
    if [ -x "$BENCHMARK_PATH" ]; then
        echo -e "\n${GREEN}--- Running $BENCHMARK ---${NC}"
        "$BENCHMARK_PATH" "$@"
    else
        echo -e "${YELLOW}Warning: Benchmark executable '$BENCHMARK' not found at $BENCHMARK_PATH. Did it compile successfully?${NC}"
    fi
done

echo -e "\n${GREEN}=== Benchmarks Completed ===${NC}"
