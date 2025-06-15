#!/bin/bash

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This profiling script is designed for macOS and uses the 'sample' command."
    echo "On Linux, you could use 'perf'. On Windows, 'VTune' or other profilers."
    exit 1
fi

set -e

# Parse arguments
BENCH_TYPE="both"
if [[ $# -gt 0 ]]; then
    case $1 in
        eval|equity|both)
            BENCH_TYPE=$1
            ;;
        *)
            echo "Usage: $0 [eval|equity|both]"
            echo "  eval   - Profile evaluation benchmark only"
            echo "  equity - Profile equity benchmark only"
            echo "  both   - Profile both benchmarks (default)"
            exit 1
            ;;
    esac
fi

# Build benchmark first (debug mode for better profiling)
echo "Building benchmark..."
zig build bench

# Find the benchmark executable
BENCH_EXE="./zig-out/bin/benchmark"
if [[ ! -f "$BENCH_EXE" ]]; then
    echo "Error: Benchmark executable not found at $BENCH_EXE"
    exit 1
fi

echo "Profiling $BENCH_TYPE benchmark(s) with sample ..."

# Start benchmark in background and profile it
$BENCH_EXE $BENCH_TYPE &
BENCH_PID=$!

# Sample the running process for 10 seconds with full paths
sample $BENCH_PID 10 1 -fullPaths -file profile_output.txt

# Wait for benchmark to finish
wait $BENCH_PID

echo "Profile saved to profile_output.txt"
echo ""
echo "Top functions by sample count:"
grep -E "^\s+[0-9]+" profile_output.txt | head -30

echo ""
echo "Full call tree (more detail):"
awk '/Call graph:/{flag=1; next} /^Total number in stack/{flag=0} flag' profile_output.txt | head -50
