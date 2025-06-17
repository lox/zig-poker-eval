#!/bin/bash

# Performance regression check script
# Compares current performance against baseline using git worktree

set -e

BASELINE_COMMIT="${1:-HEAD}"
REGRESSION_THRESHOLD="${2:-5.0}"  # 5% regression threshold
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKTREE_DIR=$(mktemp -d)

cleanup() {
    cd "$PROJECT_ROOT"
    rm -f bench_results.json baseline.json current.json
    if [ -d "$WORKTREE_DIR" ]; then
        git worktree remove "$WORKTREE_DIR" --force 2>/dev/null || true
        rm -rf "$WORKTREE_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Performance Regression Check ==="
echo "Baseline: $BASELINE_COMMIT"
echo "Threshold: ${REGRESSION_THRESHOLD}% regression"
echo

cd "$PROJECT_ROOT"

# Create worktree for baseline
echo "Creating baseline worktree..."
if ! git worktree add "$WORKTREE_DIR" "$BASELINE_COMMIT" >/dev/null 2>&1; then
    echo "Failed to create worktree. Trying alternative approach..."
    mkdir -p "$WORKTREE_DIR"
    git clone . "$WORKTREE_DIR" >/dev/null 2>&1
    cd "$WORKTREE_DIR"
    git checkout "$BASELINE_COMMIT" >/dev/null 2>&1
    cd "$PROJECT_ROOT"
fi

# Build and benchmark baseline
echo "Building baseline..."
cd "$WORKTREE_DIR"
zig build -Doptimize=ReleaseFast >/dev/null
echo "Running baseline benchmark..."
zig build bench -Doptimize=ReleaseFast -- --eval --format json >/dev/null
mv bench_results.json "$PROJECT_ROOT/baseline.json"

baseline_ns=$(jq -r '.ns_per_op' "$PROJECT_ROOT/baseline.json")
echo "Baseline: ${baseline_ns} ns/op"

# Build and benchmark current
echo "Building current version..."
cd "$PROJECT_ROOT"
zig build -Doptimize=ReleaseFast >/dev/null
echo "Running current benchmark..."
zig build bench -Doptimize=ReleaseFast -- --eval --format json >/dev/null
mv bench_results.json current.json

current_ns=$(jq -r '.ns_per_op' current.json)
echo "Current: ${current_ns} ns/op"

# Calculate regression percentage using jq
regression=$(jq -n --argjson current "$current_ns" --argjson baseline "$baseline_ns" \
    '($current - $baseline) / $baseline * 100 | . * 100 | round / 100')

echo "Performance change: ${regression}%"

# Check if regression exceeds threshold
if jq -n --argjson reg "$regression" --argjson thresh "$REGRESSION_THRESHOLD" '$reg > $thresh' | grep -q true; then
    echo "❌ PERFORMANCE REGRESSION DETECTED!"
    echo "Current performance is ${regression}% slower than baseline"
    echo "Threshold: ${REGRESSION_THRESHOLD}%"
    exit 1
elif jq -n --argjson reg "$regression" '$reg < -2.0' | grep -q true; then
    echo "🚀 PERFORMANCE IMPROVEMENT: ${regression}% faster!"
else
    echo "✅ Performance within acceptable range"
fi

echo "Performance regression check passed"
