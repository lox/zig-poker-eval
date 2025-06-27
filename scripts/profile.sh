#!/bin/bash

# Unified profiling script for poker hand evaluator
# Combines the best features of profile_bench.sh and simple_profile.sh

# Check for macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This profiling script is designed for macOS and uses the 'sample' command."
    echo "On Linux, you could use 'perf'. On Windows, 'VTune' or other profilers."
    exit 1
fi

set -e

# Parse arguments
ITERATIONS=${1:-20000000}  # Default 20M iterations
DURATION=${2:-15}          # Default 15 seconds sampling
OUTPUT_FILE=${3:-profile_output.txt}

echo "=== Poker Evaluator Profiler ==="
echo "Iterations: $ITERATIONS"
echo "Sample duration: ${DURATION}s"
echo "Output file: $OUTPUT_FILE"
echo ""

# Build benchmark with debug symbols
echo "Building benchmark..."
zig build bench -Doptimize=ReleaseFast -Dcpu=native 

# Find the executable
BENCH_EXE=$(find .zig-cache -name "bench" -type f | head -1)

if [[ ! -f "$BENCH_EXE" ]]; then
    echo "Error: Benchmark executable not found in .zig-cache"
    exit 1
fi

echo "Found benchmark: $BENCH_EXE"
echo "Starting benchmark with $ITERATIONS iterations..."

# Run benchmark in background
$BENCH_EXE --iterations $ITERATIONS &
BENCH_PID=$!

sleep 0.5  # Give benchmark time to start

echo "Profiling PID $BENCH_PID for ${DURATION}s (sampling every 1ms)..."
sample $BENCH_PID $DURATION 1 -fullPaths -file "$OUTPUT_FILE"

# Wait for benchmark to complete
wait $BENCH_PID

echo ""
echo "=== PROFILING RESULTS ==="
echo "Profile saved to: $OUTPUT_FILE"

# Count total samples
TOTAL_SAMPLES=$(grep -c 'Thread_' "$OUTPUT_FILE" 2>/dev/null || echo "0")
echo "Total samples: $TOTAL_SAMPLES"

if [[ $TOTAL_SAMPLES -eq 0 ]]; then
    echo "❌ No samples collected - benchmark may have finished too quickly"
    echo "Try increasing iterations or reducing sample duration"
    exit 1
fi

echo ""
echo "Top functions by sample count:"
grep -E "^\s+[0-9]+" "$OUTPUT_FILE" | head -10

echo ""
echo "Function breakdown:"
awk '/Sort by top of stack/{flag=1; next} /Binary Images/{flag=0} flag' "$OUTPUT_FILE"

echo ""
echo "📋 Usage: $0 [iterations] [duration] [output_file]"
echo "📁 View full profile: cat $OUTPUT_FILE"