# Performance Guide

## Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks and compare with baseline
poker-eval bench

# Run specific benchmark suite
poker-eval bench --filter eval      # Hand evaluation only
poker-eval bench --filter showdown  # Showdown benchmarks only
poker-eval bench --filter equity    # Equity calculations only

# Save new baseline (requires CV < 5%)
poker-eval bench --baseline

# Use custom regression threshold (default 5%)
poker-eval bench --threshold 10.0

# Build with optimizations first
zig build -Doptimize=ReleaseFast
```

**Available Suites:**
- `eval` - Hand evaluation (batch processing)
- `showdown` - Showdown comparison (context vs batched)
- `equity` - Equity calculations (Monte Carlo & exact)
- `range` - Range vs range equity

**Output Example:**
```
🚀 Running Benchmarks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Build mode: ReleaseFast
Version:    v2.9.0-3-g6a1e5ae69524

[1/4] eval
  • batch_evaluation: 3.04 ns/hand (329.62M/s)
[2/4] showdown
  • context_path: 30.13 ns/eval (33.19M/s)
  • batched: 7.22 ns/eval (138.52M/s)
[3/4] equity
  • monte_carlo: 439.69 µs/calc (2.27/s)
  • exact_turn: 4.30 µs/calc (231.85/s)
[4/4] range
  • equity_monte_carlo: 123.30 ms/calc (0.01/s)

📊 Comparison vs Baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ No significant changes

✅ PASSED
```

### Results (Apple M1)

**Single vs Batch:**
```
Batch Size | ns/hand | Million hands/sec | Speedup
-----------|---------|-------------------|--------
         1 |    8.67 |             115.4 |   1.00x
        32 |    4.46 |             224.3 |   1.94x
```

**Batch Size Scaling:**
```
Batch Size | ns/hand | Million hands/sec | Speedup
-----------|---------|-------------------|--------
         2 |    7.22 |             138.5 |   1.20x
         4 |    6.07 |             164.8 |   1.43x
         8 |    5.44 |             183.8 |   1.59x
        16 |    4.85 |             206.2 |   1.79x
        32 |    4.46 |             224.3 |   1.94x
        64 |    4.31 |             232.0 |   2.01x
```

**Showdown Benchmark:**
```
Scenario         | ns/eval | Comparisons/sec | Speedup
-----------------|---------|-----------------|--------
Context path     |  41.92  |       23.9M     |  1.00x
Batched (32 max) |  13.42  |       74.5M     |  3.12x
```

**Equity Calculations:**
```
Scenario           | Simulations | Time (ms) | Sims/sec (M)
-------------------|-------------|-----------|-------------
Preflop (no board) |      10000  |     12.45 |        0.803
Flop (3 cards)     |       5000  |      8.23 |        0.608
Turn (4 cards)     |       2000  |      3.89 |        0.514
River (5 cards)    |       1000  |      2.11 |        0.474
```

### Benchmark Methodology

**Stability Features:**
- **IQM Statistics**: Drops top/bottom 25% of samples to eliminate outliers
- **Repeated Measurements**: Same dataset measured multiple times for longer timing windows
- **Median as Primary Metric**: More robust to skew than mean
- **Coefficient of Variation (CV)**: Warns if CV ≥ 5% (unstable measurement)

**Configuration:**
- Warmup: 3 runs to stabilize CPU frequency and cache state
- Normal mode: 10 runs with IQM statistics
- Baseline mode: 100 runs for maximum confidence
- Build mode isolation: Separate baselines per optimization level

**Per-Suite Parameters:**
- `eval/batch_evaluation`: 100K iterations × 32 hands = 3.2M evaluations per run
- `showdown/context_path`: 100K cases × 10 repeats = 1M evaluations
- `showdown/batched`: 100K cases × 10 repeats = 1M evaluations
- `equity/monte_carlo`: 1K iterations × 10K simulations
- `equity/exact_turn`: 100 iterations × 50 repeats × 44 rivers
- `range/equity_monte_carlo`: 10 iterations × 10 repeats × 1K simulations

**Baseline Management:**
- Baselines stored in `benchmark-baseline-{mode}.json`
- Automatically selects correct baseline for build mode
- Refuses to save if any benchmark has CV ≥ 5%
- Includes system info for cross-platform comparison warnings

## Profiling

### Setup

```bash
cargo install uniprof
```

### Running Profiles

```bash
# Profile hand evaluation
task profile:eval

# Profile equity calculations
task profile:equity

# Profile showdown evaluation
task profile:showdown

# Custom iteration count
task profile:eval ITERATIONS=50000000

# Custom output directory
task profile:eval PROFILE_DIR=/tmp/my_profile
```

### Analyzing Results

**Terminal analysis:**
```bash
uniprof analyze /tmp/eval_profile/profile.json
```

Shows function call tree, time percentages, self vs total time, call counts.

**Browser visualization:**
```bash
uniprof visualize /tmp/eval_profile/profile.json
```

Interactive flamegraph with search, zoom, and export.

### Key Metrics

- **Self Time**: Time in function itself (excluding callees)
- **Total Time**: Time including all function calls
- **Call Count**: Number of invocations

### What to Look For

1. Wide bars in flamegraph - significant time consumption
2. Tall stacks - deep call chains (inlining opportunities)
3. Unexpected hotspots - algorithmic issues
4. Memory operations - cache misses or allocation overhead

## Optimization Workflow

1. **Establish baseline:**
   ```bash
   poker-eval bench --baseline
   ```

2. **Profile to find bottlenecks:**
   ```bash
   task profile:eval
   uniprof visualize /tmp/eval_profile/profile.json
   ```

3. **Identify targets:** Functions > 20% of total time

4. **Optimize:** SIMD vectorization, loop unrolling, cache-friendly layout

5. **Benchmark improvement:**
   ```bash
   poker-eval bench
   ```

6. **Re-profile to verify:**
   ```bash
   task profile:eval
   ```

7. **Repeat until target achieved**

### Comparing Optimizations

```bash
# Before
task profile:eval PROFILE_DIR=/tmp/before
cp /tmp/before/profile.json /tmp/before.json

# Make changes, rebuild

# After
task profile:eval PROFILE_DIR=/tmp/after
cp /tmp/after/profile.json /tmp/after.json

# Compare
uniprof visualize /tmp/before.json
uniprof visualize /tmp/after.json
```

## Build Flags

```bash
# Maximum performance (required for accurate benchmarks)
-Doptimize=ReleaseFast

# Native CPU features
-Dcpu=native

# Frame pointers for profiling (Debug/ReleaseSafe only)
# ReleaseFast omits frame pointers for max performance
```

## Platform Notes

**macOS:**
- Uniprof uses native Instruments sampling
- No special performance modes needed on Apple Silicon
- Thermal throttling rarely an issue with M1 efficiency

**Linux:**
- Uniprof uses `perf`
- Check availability: `perf --version`
- Adjust permissions: `echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid`

**Windows:**
- Uniprof uses ETW (Event Tracing for Windows)
- Run as administrator

## Performance Tips

1. Close heavy applications (browsers, IDEs)
2. Ensure adequate cooling for sustained benchmarks
3. Use Activity Monitor to verify no CPU-intensive background tasks
4. Always use `-Doptimize=ReleaseFast` for performance testing

## Troubleshooting

**High coefficient of variation (> 5%):**
- Cause: System interference, thermal issues
- Solution: Fewer background processes
- Check: `pmset -g thermlog` for thermal state

**Performance below target (~3.3ns/hand on M1):**
- Wrong build flags (not using ReleaseFast)
- Debug build accidentally used
- Flush-heavy test data (should be < 0.4%)

**Validation failures (accuracy < 100%):**
- Table corruption
- Build configuration mismatch
- Solution: `zig build build-tables -Doptimize=ReleaseFast`

**Missing symbols in profile:**
- Solution: `dsymutil zig-out/bin/poker-eval`

**Permission errors:**
- macOS: Allow in System Preferences > Privacy
- Linux: Adjust `perf_event_paranoid`
- Windows: Run as administrator

## Custom Profiling

```bash
# Build with frame pointers
task build

# Generate debug symbols
dsymutil zig-out/bin/poker-eval

# Profile custom command
uniprof record --platform native -o my_profile.json -- \
  zig-out/bin/poker-eval equity "AhAs" "KdKc" --sims 100000
```

## Reference

See [experiments.md](experiments.md) for detailed optimization experiments and lessons learned.
