#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color
RED='\033[0;31m'
YELLOW='\033[1;33m'

echo -e "${GREEN}=== HELIX Build Script ===${NC}"

# Check if CMake is installed
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is not installed or not in PATH.${NC}" >&2
    exit 1
fi

# Parse options
CLEAN_BUILD=false
BUILD_TYPE="Debug"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clean) CLEAN_BUILD=true ;;
        --release) BUILD_TYPE="Release" ;;
        --debug) BUILD_TYPE="Debug" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Path to workspace root
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKSPACE_DIR"

# Handle clean build
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning previous build directories...${NC}"
    rm -rf out build
fi

# Check if Preset is available and use it, otherwise fall back to standard configure
if [ -f "CMakePresets.json" ]; then
    echo -e "${GREEN}Configuring with CMake Preset 'HELIX' (${BUILD_TYPE})...${NC}"
    cmake --preset HELIX -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    echo -e "${GREEN}Building the project...${NC}"
    cmake --build out/build/HELIX
else
    echo -e "${YELLOW}CMakePresets.json not found. Falling back to standard CMake build...${NC}"
    echo -e "${GREEN}Configuring build directory (${BUILD_TYPE})...${NC}"
    cmake -B build -S . -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    
    echo -e "${GREEN}Building the project...${NC}"
    cmake --build build
fi

echo -e "${GREEN}Build completed successfully!${NC}"
