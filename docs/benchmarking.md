# Benchmarking Guide

This guide describes the benchmarking methodology used for the Zig poker evaluator, which achieves ~4.5ns per hand evaluation on Apple M1.

## 1. Benchmark Implementation

The benchmarking framework is implemented in `src/tools/benchmark.zig` with these key features:

### Core Benchmarking Options

```zig
pub const BenchmarkOptions = struct {
    iterations: u32 = 100000,          // Number of batches to evaluate
    warmup: bool = true,               // Warm up caches before timing
    measure_overhead: bool = true,     // Measure framework overhead
    multiple_runs: bool = true,        // Run 5 times for statistics
    show_comparison: bool = true,      // Compare batch vs single-hand
    verbose: bool = false,
};
```

### Batch Processing

- **Default batch size**: 32 hands (optimal for amortizing overhead)
- **Test corpus**: 1.6M randomly generated hands to ensure cache pressure
- **Fixed seed**: 42 for reproducible results

## 2. Running Benchmarks

### Basic Benchmark

```bash
# Run with default options (100K iterations × 32 hands = 3.2M hands)
task bench:eval

# With custom iterations
task bench:eval -- --iterations 1000000
```

### Quick Validation

```bash
# Quick benchmark with correctness validation
task bench:eval -- --quick --validate
```

### Benchmark Types

```bash
# Hand evaluation performance
task bench:eval

# Equity calculation performance
task bench:equity

# Showdown evaluation (scalar vs batched)
task bench:showdown

# Test different batch sizes (2, 4, 8, 16, 32, 64)
task bench:eval -- --batch-sizes
```

## 3. Measurement Methodology

### Cache Warmup

The benchmark performs cache warmup by:

1. Evaluating 1024 random hands to touch lookup tables
2. Processing first 64K test hands in batches
3. Ensuring tables are L2-resident before timing

### Overhead Measurement

Framework overhead is measured using a dummy evaluator that only performs `@popCount`:

```zig
fn benchmarkDummyEvaluator(iterations: u32, test_hands: []const u64) f64 {
    // Measures loop overhead, memory access, and timing overhead
    checksum +%= @popCount(test_hands[(hand_idx + j) % test_hands.len]);
}
```

### Statistical Analysis

- **Multiple runs**: 5 independent runs with median selection
- **Coefficient of variation**: Calculated to ensure measurement quality
- **Overhead correction**: Subtracts framework overhead from final times

### Timer Implementation

- Uses `std.time.nanoTimestamp()` which wraps `mach_absolute_time()` on macOS
- Measures at batch boundaries to amortize timer overhead
- ~40ns timer overhead per measurement call

## 4. Performance Results (Apple M1)

### Single vs Batch Performance

```text
Batch Size | ns/hand | Million hands/sec | Speedup vs single
-----------|---------|-------------------|------------------
         1 |    8.67 |             115.4 |             1.00x
        32 |    4.46 |             224.3 |             1.94x
```

### Batch Size Scaling

```text
Batch Size | ns/hand | Million hands/sec | Speedup vs single
-----------|---------|-------------------|------------------
         2 |    7.22 |             138.5 |             1.20x
         4 |    6.07 |             164.8 |             1.43x
         8 |    5.44 |             183.8 |             1.59x
        16 |    4.85 |             206.2 |             1.79x
        32 |    4.46 |             224.3 |             1.94x
        64 |    4.31 |             232.0 |             2.01x
```

### Showdown Benchmark (BoardContext)

Context-aware batching now measures directly via `poker-eval bench --showdown`. On Apple M1 with 320K hero/villain comparisons sharing the same board, batching delivers a ~3.1× speedup over the context-only path:

```text
Scenario         | ns/eval | Comparisons/sec | Speedup vs context
-----------------|---------|-----------------|-------------------
Context path     |  41.92  |       23.9M     | 1.00x
Batched (32 max) |  13.42  |       74.5M     | 3.12x
```

The benchmark generates board-coherent workloads (groups of up to 32 hero/villain pairs per board), reuses `BoardContext`, and feeds each chunk into the SIMD evaluator. Use `--iterations N` to scale the sample size; each iteration represents one hero/villain pair.

## 5. Correctness Validation

### Automatic Validation

The benchmark includes built-in correctness checking:

```zig
pub fn validateCorrectness(test_hands: []const u64) !bool {
    // Validates first 16K hands against slow reference implementation
    // Requires 100% accuracy - any mismatch returns error
    const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total));
    if (accuracy < 1.0) {
        return error.AccuracyTooLow;
    }
}
```

### Test Coverage

- **Basic validation**: 16K hands (512 batches of 32)
- **Edge cases**: Two trips → full house, wheel straights
- **Accuracy requirement**: 100% match with reference implementation
- **XOR checksum**: Rolling checksum prevents silent errors

## 6. Equity Benchmarks

The framework also benchmarks Monte Carlo equity calculations:

### Head-to-Head Equity

```text
Scenario           | Simulations | Time (ms) | Sims/sec (millions)
-------------------|-------------|-----------|--------------------
Preflop (no board) |      10000  |     12.45 |              0.803
Flop (3 cards)     |       5000  |      8.23 |              0.608
Turn (4 cards)     |       2000  |      3.89 |              0.514
River (5 cards)    |       1000  |      2.11 |              0.474
```

### Multi-Way Equity

```text
3 players          |       5000  |     18.67 |              0.268
4 players          |       5000  |     24.89 |              0.201
6 players          |       5000  |     37.45 |              0.134
9 players          |       5000  |     56.78 |              0.088
```

## 7. Hardware and OS Setup

### macOS (Primary Platform)

- No special performance modes needed on Apple Silicon
- Thermal throttling rarely an issue with M1's efficiency
- Background process impact minimal due to efficiency cores

### Performance Consistency Tips

1. Close heavy applications (browsers, IDEs)
2. Ensure adequate cooling for sustained benchmarks
3. Use Activity Monitor to verify no CPU-intensive background tasks

### Build Environment

```bash
# Essential flags for performance
-Doptimize=ReleaseFast    # Maximum optimization
-Dcpu=native              # Use all CPU features

# Optional: Disable safety checks (already done in ReleaseFast)
# -Drelease-safe=false
```

## 8. Key Insights from Benchmarking

### What Works

1. **SIMD batch processing**: ~2x speedup over single-hand evaluation
2. **Structure-of-arrays layout**: Enables efficient vectorization
3. **Cache-resident tables**: 267KB fits in L2, no memory stalls
4. **Fixed batch size 32**: Optimal balance of register usage and overhead

### What Doesn't Work

From benchmarking experiments:

1. **Batch sizes > 32**: Diminishing returns, register pressure
2. **Prefetching**: Tables already cache-resident
3. **Complex overhead correction**: Simple popcount baseline sufficient

## 9. Troubleshooting

### High Coefficient of Variation (> 5%)

- **Cause**: System interference, thermal issues
- **Solution**: Run with fewer background processes
- **Check**: `pmset -g thermlog` for thermal state

### Performance Below Target

- **Expected**: ~4.5ns per hand on M1
- **Common issues**:
  - Wrong build flags (not using ReleaseFast)
  - Debug build accidentally used
  - Flush-heavy test data (should be < 0.4%)

### Validation Failures

- **Symptoms**: Accuracy < 100%
- **Causes**:
  - Table corruption
  - Build configuration mismatch
- **Solution**: `zig build build-tables -Doptimize=ReleaseFast`

## 10. Advanced Benchmarking

### Custom Test Hands

```zig
// Test specific hand patterns
pub fn testSingleHand(hand: u64) struct { slow: u16, fast: u16, match: bool } {
    const slow_result = poker.slow.evaluateHand(hand);
    const fast_result = poker.evaluateHand(hand);
    return .{
        .slow = slow_result,
        .fast = fast_result,
        .match = slow_result == fast_result,
    };
}
```

### Profile-Guided Analysis

See `profiling.md` for detailed profiling instructions to identify bottlenecks beyond timing measurements.
