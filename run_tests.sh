#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color
RED='\033[0;31m'
YELLOW='\033[1;33m'

echo -e "${GREEN}=== HELIX Test Runner ===${NC}"

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

echo -e "${GREEN}Running unit tests in: ${BUILD_DIR}${NC}"

# Output unit tests to JSON in output directory
mkdir -p "$WORKSPACE_DIR/output/unit_tests"
export GTEST_OUTPUT="json:$WORKSPACE_DIR/output/unit_tests/"

# Run the tests using ctest
# Pass any arguments received to ctest (e.g. -R to filter tests)
ctest --test-dir "$BUILD_DIR" --output-on-failure "$@"
