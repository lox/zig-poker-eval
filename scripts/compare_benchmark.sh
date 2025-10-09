#!/bin/bash
set -euo pipefail

# Compare benchmark results and fail if performance regressed
# Usage: compare_benchmark.sh <baseline.json> <current.json>

# Check dependencies
for cmd in jq awk; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: $cmd is required but not installed"
    exit 1
  fi
done

BASELINE_FILE="${1:-}"
CURRENT_FILE="${2:-}"
THRESHOLD="${THRESHOLD:-10}"  # 10% regression threshold by default

if [[ -z "$BASELINE_FILE" ]] || [[ -z "$CURRENT_FILE" ]]; then
  echo "Usage: $0 <baseline.json> <current.json>"
  echo ""
  echo "Environment variables:"
  echo "  THRESHOLD - Regression threshold percentage (default: 10)"
  exit 1
fi

if [[ ! -f "$BASELINE_FILE" ]]; then
  echo "Error: Baseline file not found: $BASELINE_FILE"
  exit 1
fi

if [[ ! -f "$CURRENT_FILE" ]]; then
  echo "Error: Current results file not found: $CURRENT_FILE"
  exit 1
fi

# Extract metrics using jq
baseline_ns=$(jq -r '.ns_per_hand' "$BASELINE_FILE")
current_ns=$(jq -r '.ns_per_hand' "$CURRENT_FILE")

if [[ -z "$baseline_ns" ]] || [[ "$baseline_ns" == "null" ]]; then
  echo "Warning: Could not extract ns_per_hand from baseline file"
  echo "Baseline file contents:"
  cat "$BASELINE_FILE"
  echo ""
  echo "Skipping comparison - baseline may be from incompatible version"
  echo "Current results:"
  cat "$CURRENT_FILE"
  exit 0
fi

if [[ -z "$current_ns" ]] || [[ "$current_ns" == "null" ]]; then
  echo "Error: Could not extract ns_per_hand from current file"
  echo "Current file contents:"
  cat "$CURRENT_FILE"
  exit 1
fi

baseline_hps=$(jq -r '.hands_per_second' "$BASELINE_FILE")
current_hps=$(jq -r '.hands_per_second' "$CURRENT_FILE")

# Calculate percentage change
# Positive change = regression (slower), negative = improvement (faster)
change_pct=$(awk "BEGIN {printf \"%.2f\", (($current_ns - $baseline_ns) / $baseline_ns) * 100}")

echo "Benchmark Comparison"
echo "===================="
echo ""
echo "Baseline:  ${baseline_ns} ns/hand (${baseline_hps} hands/sec)"
echo "Current:   ${current_ns} ns/hand (${current_hps} hands/sec)"
echo ""

# Check if we have a regression
if awk "BEGIN {exit !($change_pct > $THRESHOLD)}"; then
  echo "❌ REGRESSION DETECTED: Performance decreased by ${change_pct}%"
  echo "   Threshold: ${THRESHOLD}%"
  exit 1
elif awk "BEGIN {exit !($change_pct > 0)}"; then
  echo "⚠️  Minor slowdown: ${change_pct}% (within ${THRESHOLD}% threshold)"
  exit 0
else
  # Improvement
  improvement=$(awk "BEGIN {printf \"%.2f\", -1 * $change_pct}")
  echo "✅ Performance improved by ${improvement}%"
  exit 0
fi
