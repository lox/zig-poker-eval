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
Version:    v3.6.1-0-g25f53880dfc3
Mode:       Baseline (100 runs per benchmark)

[1/6] eval
  • batch_evaluation: 2.00 ns/hand (500.52M/s)
  • single_evaluation: 4.89 ns/hand (204.71M/s)
[2/6] context
  • init_board: 6.38 ns/call (6.38 ns/call)
  • hole_evaluation: 4.97 ns/eval (201.02M/s)
[3/6] showdown
  • context_path: 11.01 ns/eval (90.86M/s)
  • batched: 4.54 ns/eval (220.30M/s)
[4/6] multiway
  • showdown_multiway: 24.86 ns/eval (40.22M/s)
  • equity_weights: 34.14 ns/eval (29.29M/s)
[5/6] equity
  • monte_carlo: 282.02 µs/calc (3.55K/s)
  • exact_turn: 0.20 µs/calc (4.99M/s)
[6/6] range
  • equity_monte_carlo: 0.08 ms/calc (11.94K/s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Baseline saved to benchmark-baseline-release-fast.json
```

### Results (MacBook Pro M5)

All measurements below use `zig build bench --baseline` on October 30, 2025 with `-Doptimize=ReleaseFast`. Throughput conversions use 1,000/ns and round to the nearest integer.

**Evaluator**

| Metric | Time (ns/hand) | Throughput |
| --- | --- | --- |
| Single evaluation (scalar) | 4.89 | 205M hands/s |
| Batch evaluation (32 hands) | 2.00 | 501M hands/s |

**Context Setup**

| Metric | Time | Throughput |
| --- | --- | --- |
| init_board | 6.38 ns/call | 157M calls/s |
| hole_evaluation | 4.97 ns/eval | 201M eval/s |

**Showdown**

| Metric | Time | Throughput |
| --- | --- | --- |
| context_path | 11.01 ns/eval | 91M eval/s |
| batched (32 lanes) | 4.54 ns/eval | 220M eval/s |

**Multiway**

| Metric | Time | Throughput |
| --- | --- | --- |
| showdown_multiway | 24.86 ns/eval | 40M eval/s |
| equity_weights | 34.14 ns/eval | 29M eval/s |

**Equity and Range**

| Metric | Time | Throughput |
| --- | --- | --- |
| monte_carlo | 282.02 µs/calc | 3.55K calc/s |
| exact_turn | 0.20 µs/calc | 5.0M calc/s |
| range equity_monte_carlo | 0.0837 ms/calc | 11.9K calc/s |

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
npm install -g uniprof
```

### Running Profiles

```bash
# Profile hot paths (pokerbot workloads)
task profile                                    # Default: showdown scenario
task profile SCENARIO=init-board                # Board context initialization
task profile SCENARIO=showdown                  # Showdown evaluation
task profile SCENARIO=multiway                  # Multiway hand evaluation

# Custom iteration count
task profile SCENARIO=showdown ITERATIONS=100000000
```

**Available scenarios:**
- `init-board` - Board context initialization
- `showdown` - Showdown evaluation (heads-up comparison)
- `multiway` - Single hand evaluation (6-max equity calculation)

**Output location:** `/tmp/profile_<scenario>/profile.json`

**Example workflow:**
```bash
# Profile showdown hot path
task profile SCENARIO=showdown

# Analyze results
uniprof analyze /tmp/profile_showdown/profile.json

# Visualize in browser
uniprof visualize /tmp/profile_showdown/profile.json
```

### Analyzing Results

**Terminal analysis:**
```bash
uniprof analyze /tmp/profile_showdown/profile.json
```

Shows function call tree, time percentages, self vs total time, call counts.

**Browser visualization:**
```bash
uniprof visualize /tmp/profile_showdown/profile.json
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
   task profile SCENARIO=showdown
   uniprof visualize /tmp/profile_showdown/profile.json
   ```

3. **Identify targets:** Functions > 20% of total time

4. **Optimize:** SIMD vectorization, loop unrolling, cache-friendly layout

5. **Benchmark improvement:**
   ```bash
   poker-eval bench
   ```

6. **Re-profile to verify:**
   ```bash
   task profile SCENARIO=showdown
   ```

7. **Repeat until target achieved**

### Comparing Optimizations

```bash
# Before
task profile SCENARIO=showdown
cp /tmp/profile_showdown/profile.json /tmp/before.json

# Make changes, rebuild

# After
task profile SCENARIO=showdown
cp /tmp/profile_showdown/profile.json /tmp/after.json

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
- Thermal throttling rarely an issue with M5 efficiency cores

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

**Performance below target (~2.0ns/hand on MacBook Pro M5):**
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
