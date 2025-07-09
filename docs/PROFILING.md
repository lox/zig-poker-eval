# Profiling Guide

This guide describes how to profile the Zig poker evaluator to identify performance bottlenecks and optimization opportunities.

## 1. Profiling Script

The main profiling tool is `scripts/profile.sh`, a unified script for macOS that provides high-frequency sampling with configurable parameters.

### Basic Usage
```bash
# Quick profile (default: 20M iterations, 15s sampling)
./scripts/profile.sh

# Custom parameters
./scripts/profile.sh [iterations] [duration] [output_file]

# Examples
./scripts/profile.sh 5000000 10 quick_profile.txt     # 5M iterations, 10s
./scripts/profile.sh 50000000 30 detailed_profile.txt # 50M iterations, 30s
```

### Script Features
- Builds benchmark with ReleaseFast + debug symbols
- Configurable iteration count and sampling duration
- Samples every 1ms for high resolution
- Outputs detailed function-level breakdown
- Error handling for failed sampling

## 2. Understanding Profile Output

### Sample Output
```
=== PROFILING RESULTS ===
Total samples: 3886

Function breakdown:
    evaluator.computeRpcFromHand         3815 (98.2%)
    evaluator.evaluateHand                 71  (1.8%)
```

This clearly shows that 98.2% of time is spent in RPC computation (rank counting), which led to the SIMD optimization efforts.

### Key Metrics
- **Total samples**: Higher is better for statistical significance
- **Function breakdown**: Shows percentage of time in each function
- **Line-level hotspots**: Identifies specific code lines consuming CPU

## 3. Platform-Specific Profiling

### macOS (Primary Platform)
Uses the `sample` command:
```bash
sample <PID> <duration> <interval_ms> -fullPaths -file output.txt
```

### Linux Alternative
```bash
# Use perf instead
perf record -g -F 1000 ./zig-out/bin/poker-eval bench
perf report
```

### Windows Alternative
- Use Intel VTune or Visual Studio Profiler
- AMD uProf for AMD processors

## 4. Common Profiling Scenarios

### Finding Bottlenecks
```bash
# Long profile for detailed analysis
./scripts/profile.sh 100000000 30 bottleneck_analysis.txt

# Look for functions > 10% of samples
grep -E "^\s+[0-9]+" bottleneck_analysis.txt | head -20
```

### Comparing Optimizations
```bash
# Profile before optimization
./scripts/profile.sh 20000000 15 before.txt

# Make changes, rebuild, profile after
./scripts/profile.sh 20000000 15 after.txt

# Compare function percentages
```

### SIMD Effectiveness
Profile different batch sizes to see vectorization benefits:
```bash
# Profile single-hand evaluation
zig build run -Doptimize=ReleaseFast -- bench --batch-size 1 &
BENCH_PID=$!
sample $BENCH_PID 10 1 -file single_hand.txt

# Profile batch evaluation
zig build run -Doptimize=ReleaseFast -- bench --batch-size 32 &
BENCH_PID=$!
sample $BENCH_PID 10 1 -file batch_32.txt
```

## 5. Key Findings from Profiling

### Before SIMD Optimization
```
computeRpcFromHand    98.2%  # Rank counting dominates
evaluateHand           1.8%  # Table lookup negligible
```

### After SIMD Optimization
```
computeRpcSimd        45.3%  # Vectorized rank counting
evaluateBatch         38.2%  # Batch coordination overhead
chdLookupScalar       16.5%  # Table lookups now visible
```

The SIMD optimization successfully reduced RPC computation time by ~50%, making other operations visible in the profile.

## 6. Advanced Profiling Techniques

### Sampling Frequency
- **1ms interval**: Good balance of resolution and overhead
- **0.1ms interval**: Maximum resolution but higher overhead
- **10ms interval**: Low overhead for long-running benchmarks

### Profile-Guided Optimization
1. Profile to find hotspots
2. Optimize the top function
3. Re-profile to verify improvement
4. Repeat until diminishing returns

### Cache Analysis
While `sample` doesn't show cache misses directly, you can infer cache behavior:
- Consistent timing = good cache behavior
- High variation = possible cache misses

## 7. Troubleshooting Profiling Issues

### No Samples Collected
```
❌ No samples collected - benchmark may have finished too quickly
```
**Solution**: Increase iterations or reduce sample duration

### Low Sample Count
- **Cause**: Benchmark too short for sampling interval
- **Solution**: Use more iterations (50M+ for 15s profiling)

### Missing Debug Symbols
- **Symptom**: Function names show as addresses
- **Solution**: Ensure `-Doptimize=ReleaseFast` includes debug info

## 8. Interpreting Results for Optimization

### What to Look For
1. **Functions > 20% of samples**: Primary optimization targets
2. **Unexpected hotspots**: May indicate algorithmic issues
3. **Even distribution**: Good - no single bottleneck
4. **Memory operations**: High % may indicate cache issues

### Example Analysis
From actual profiling that led to SIMD optimization:
```
computeRpcFromHand (98.2%) breakdown:
- Nested loops: 85% (13 ranks × 4 suits)
- Base-5 encoding: 13%
- Function overhead: 2%
```

This profile directly motivated the SIMD batch processing approach, which processes multiple hands in parallel through the expensive rank counting loops.

## 9. Continuous Profiling

For ongoing optimization:
1. Save baseline profiles with each release
2. Compare profiles across versions
3. Track performance metrics over time
4. Profile on different hardware (M1, M2, Intel)

## 10. Quick Reference

### Essential Commands
```bash
# Basic profile
./scripts/profile.sh

# Detailed profile
./scripts/profile.sh 50000000 30 detailed.txt

# View results
cat detailed.txt | grep -A 20 "Function breakdown"

# Find hot functions
awk '/Sort by top of stack/{flag=1; next} /Binary Images/{flag=0} flag' detailed.txt
```

### Optimization Workflow
1. Benchmark to establish baseline
2. Profile to find bottlenecks
3. Optimize identified functions
4. Benchmark to measure improvement
5. Profile to find new bottlenecks
6. Repeat until target performance achieved
