#!/bin/bash

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This profiling script is designed for macOS and uses the 'sample' command."
    echo "On Linux, you could use 'perf'. On Windows, 'VTune' or other profilers."
    exit 1
fi

set -e

# Build benchmark first (ReleaseFast but with debug symbols for profiling)
echo "Building and profiling benchmark with many iterations..."

# Use zig build directly to run benchmark with arguments and profile it
zig build bench -Doptimize=ReleaseFast -Dcpu=native -- --iterations 5000000 &
BENCH_PID=$!

echo "Profiling benchmark (PID: $BENCH_PID) for 20 seconds with high-frequency sampling..."

# Sample every 0.1ms (10x more frequent than default 1ms) for 20 seconds
sample $BENCH_PID 20 0.1 -fullPaths -file profile_output.txt

# Wait for benchmark to finish
wait $BENCH_PID

echo "Profile saved to profile_output.txt"
echo ""
echo "Top functions by sample count:"
grep -E "^\s+[0-9]+" profile_output.txt | head -20

echo ""
echo "Full call tree (more detail):"
awk '/Call graph:/{flag=1; next} /^Total number in stack/{flag=0} flag' profile_output.txt | head -40

echo ""
echo "To view full profile: cat profile_output.txt"