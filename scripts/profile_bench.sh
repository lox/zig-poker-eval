#!/bin/bash

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This profiling script is designed for macOS and uses the 'sample' command."
    echo "On Linux, you could use 'perf'. On Windows, 'VTune' or other profilers."
    exit 1
fi

set -e

# Parse arguments and convert to new flag format
BENCH_ARGS=""
if [[ $# -eq 0 ]]; then
    BENCH_ARGS="--eval --equity"
    BENCH_TYPE="evaluation and single-threaded equity (default)"
else
    case $1 in
        eval)
            BENCH_ARGS="--eval"
            BENCH_TYPE="evaluation"
            ;;
        equity)
            BENCH_ARGS="--equity"
            BENCH_TYPE="single-threaded equity"
            ;;
        equityThreaded)
            BENCH_ARGS="--equityThreaded"
            BENCH_TYPE="multi-threaded equity"
            ;;
        both)
            BENCH_ARGS="--eval --equity"
            BENCH_TYPE="evaluation and single-threaded equity"
            ;;
        all)
            BENCH_ARGS="--eval --equity --equityThreaded"
            BENCH_TYPE="all benchmarks"
            ;;
        *)
            echo "Usage: $0 [eval|equity|equityThreaded|both|all]"
            echo "  eval           - Profile evaluation benchmark only"
            echo "  equity         - Profile single-threaded equity benchmark only"
            echo "  equityThreaded - Profile multi-threaded equity benchmark only"
            echo "  both           - Profile evaluation and single-threaded equity"
            echo "  all            - Profile all benchmarks (default)"
            exit 1
            ;;
    esac
fi

# Build benchmark first (debug mode for better profiling)
echo "Building benchmark..."
zig build bench -Doptimize=Debug

# Find the benchmark executable
BENCH_EXE="./zig-out/bin/benchmark"
if [[ ! -f "$BENCH_EXE" ]]; then
    echo "Error: Benchmark executable not found at $BENCH_EXE"
    exit 1
fi

echo "Profiling $BENCH_TYPE with sample ..."

# Start benchmark in background and profile it
$BENCH_EXE $BENCH_ARGS &
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
