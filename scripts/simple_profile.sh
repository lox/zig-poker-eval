#!/bin/bash

set -e

echo "Building benchmark..."
zig build bench -Doptimize=ReleaseFast -Dcpu=native 

# Find the executable
BENCH_EXE=$(find .zig-cache -name "bench" -type f | head -1)

echo "Starting long-running benchmark for profiling..."
echo "Command: $BENCH_EXE --iterations 20000000"

# Run with 20M iterations (should take ~20+ seconds) and profile it
$BENCH_EXE --iterations 20000000 &
BENCH_PID=$!

sleep 0.5  # Give it time to start

echo "Profiling PID $BENCH_PID for 15 seconds, sampling every 1ms..."
sample $BENCH_PID 15 1 -fullPaths -file detailed_profile.txt

wait $BENCH_PID

echo ""
echo "=== PROFILING RESULTS ==="
echo "Total samples: $(grep -c 'Thread_' detailed_profile.txt)"
echo ""
echo "Top functions by sample count:"
grep -E "^\s+[0-9]+" detailed_profile.txt | head -10
echo ""
echo "Function breakdown:"
awk '/Sort by top of stack/{flag=1; next} /Binary Images/{flag=0} flag' detailed_profile.txt