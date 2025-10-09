# Profiling Guide

This guide describes how to profile the Zig poker evaluator using uniprof to identify performance bottlenecks and optimization opportunities.

## Prerequisites

Install uniprof:
```bash
cargo install uniprof
```

## Quick Start

```bash
# Profile hand evaluation (20M iterations)
task profile:eval

# Profile equity calculations
task profile:equity

# Profile showdown evaluation
task profile:showdown

# Analyze results in terminal
uniprof analyze /tmp/eval_profile/profile.json

# Visualize in browser
uniprof visualize /tmp/eval_profile/profile.json
```

## Profiling Tasks

### Hand Evaluation Performance

Profile the core hand evaluation engine:

```bash
# Default: 20M iterations
task profile:eval

# Custom iteration count
task profile:eval ITERATIONS=50000000

# Custom output directory
task profile:eval PROFILE_DIR=/tmp/my_profile
```

This profiles the batch evaluation path, showing:
- SIMD vectorization hotspots
- Lookup table access patterns
- Function call overhead

### Equity Calculation Performance

Profile Monte Carlo equity simulations:

```bash
task profile:equity

# Profiles multiple scenarios:
# - Head-to-head (preflop, flop, turn, river)
# - Multi-way (3, 4, 6, 9 players)
```

This profiles:
- Random hand generation
- Board enumeration
- Showdown evaluation
- Result accumulation

## Analyzing Profiles

### Terminal Analysis

Quick overview in the terminal:

```bash
uniprof analyze /tmp/poker_profile/profile.json
```

Shows:
- Function call tree
- Time percentages
- Self time vs total time
- Call counts

### Browser Visualization

Interactive flamegraph:

```bash
uniprof visualize /tmp/poker_profile/profile.json
```

Opens browser with:
- Flamegraph visualization
- Searchable function list
- Click to zoom/filter
- Export capabilities

## Understanding Results

### Key Metrics

**Self Time**: Time spent in the function itself (excluding callees)
**Total Time**: Time including all function calls
**Call Count**: Number of times function was called

### What to Look For

1. **Wide bars in flamegraph**: Functions consuming significant time
2. **Tall stacks**: Deep call chains (potential for inlining)
3. **Unexpected hotspots**: May indicate algorithmic issues
4. **Memory operations**: Cache misses or allocation overhead

## Optimization Workflow

1. **Establish baseline**
   ```bash
   task bench:eval -- --quick
   ```

2. **Profile to find bottlenecks**
   ```bash
   task profile:eval
   uniprof visualize /tmp/eval_profile/profile.json
   ```

3. **Identify optimization targets**
   - Functions > 20% of total time
   - Loops with high iteration counts
   - Unnecessary allocations

4. **Optimize identified functions**
   - SIMD vectorization
   - Loop unrolling
   - Cache-friendly data layout

5. **Benchmark improvement**
   ```bash
   task bench:eval
   ```

6. **Re-profile to verify**
   ```bash
   task profile:eval
   ```

7. **Repeat until target performance achieved**

## Advanced Usage

### Custom Profiling

Profile specific workloads directly:

```bash
# Build with frame pointers
task build

# Generate debug symbols
dsymutil zig-out/bin/poker-eval

# Profile custom command
uniprof record --platform native -o my_profile.json -- \
  zig-out/bin/poker-eval equity "AhAs" "KdKc" --sims 100000
```

### Comparing Optimizations

```bash
# Profile before changes
task profile:eval PROFILE_DIR=/tmp/before
cp /tmp/before/profile.json /tmp/before.json

# Make changes, rebuild, profile after
task profile:eval PROFILE_DIR=/tmp/after
cp /tmp/after/profile.json /tmp/after.json

# Compare in browser
uniprof visualize /tmp/before.json
uniprof visualize /tmp/after.json
```

### Frame Pointer Preservation

The build system automatically preserves frame pointers in Debug and ReleaseSafe modes for accurate profiling. For ReleaseFast (default), frame pointers may be omitted for maximum performance.

To profile ReleaseFast builds with frame pointers, modify `build.zig`:

```zig
.omit_frame_pointer = false,
```

## Platform Notes

### macOS (Primary)

Uniprof uses native instruments sampling on macOS for zero-overhead profiling.

### Linux

Uniprof uses `perf` on Linux. Ensure kernel perf events are enabled:

```bash
# Check perf availability
perf --version

# May need to adjust permissions
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

### Windows

Uniprof uses ETW (Event Tracing for Windows). Run as administrator for best results.

## Troubleshooting

### Missing Symbols

**Symptom**: Functions show as addresses instead of names

**Solution**: Ensure debug symbols are generated:
```bash
dsymutil zig-out/bin/poker-eval
```

### Short Sample Time

**Symptom**: Profile completes too quickly

**Solution**: Increase iterations:
```bash
task profile:eval ITERATIONS=100000000
```

### Permission Errors

**Symptom**: Uniprof fails to attach

**Solution**: Grant necessary permissions (varies by platform):
- macOS: Allow in System Preferences > Privacy
- Linux: Adjust perf_event_paranoid
- Windows: Run as administrator

## Quick Reference

```bash
# Common workflows
task profile:eval                     # Profile hand evaluation
task profile:equity                   # Profile equity calculations
task profile:showdown                 # Profile showdown evaluation
uniprof analyze <json>                # Terminal analysis
uniprof visualize <json>              # Browser visualization

# Benchmarking
task bench:eval                       # Evaluation benchmark
task bench:equity                     # Equity benchmark
task bench:showdown                   # Showdown benchmark

# Custom profiling
uniprof record -o out.json -- <cmd>  # Record custom command
```

## Further Reading

- [uniprof documentation](https://github.com/emmanuelantony2000/uniprof)
- Flamegraph interpretation guides
- CPU profiling best practices
