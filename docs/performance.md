# Performance Guide

## Benchmarking

### Running Benchmarks

```bash
# Hand evaluation
task bench:eval

# Equity calculations
task bench:equity

# Showdown evaluation
task bench:showdown

# Custom iterations
task bench:eval -- --iterations 1000000

# Quick validation
task bench:eval -- --quick --validate
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

### Benchmark Options

```zig
pub const BenchmarkOptions = struct {
    iterations: u32 = 100000,
    warmup: bool = true,
    measure_overhead: bool = true,
    multiple_runs: bool = true,
    show_comparison: bool = true,
    verbose: bool = false,
};
```

**Methodology:**
- Default: 100K iterations × 32 hands = 3.2M hands
- Test corpus: 1.6M random hands
- Fixed seed: 42 (reproducible results)
- Cache warmup: 1024 hands + 64K batch processing
- Multiple runs: 5 runs with median selection
- Overhead measurement: Dummy evaluator baseline

### Correctness Validation

```bash
task bench:eval -- --validate
```

Validates first 16K hands against reference implementation. Requires 100% accuracy.

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
   task bench:eval -- --quick
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
   task bench:eval
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
