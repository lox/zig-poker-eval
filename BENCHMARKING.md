# Benchmarking Methodology

## Hardware and OS Setup

1. Document platform: CPU model, macOS version, memory configuration

2. Optimize for consistent performance:
   - Set high performance mode: `sudo pmset -a highperf 1`
   - Check thermal state: `pmset -g thermlog`
   - Close unnecessary applications

3. Minimize system interference:
   - Run benchmark with elevated priority: `sudo nice -20`
   - Consider disabling Spotlight indexing temporarily

## Build Discipline

1. Build in release mode:
   ```
   zig build bench -Doptimize=ReleaseFast -Dcpu=native
   ```

2. Record build environment:
   - Zig version and build flags
   - macOS version and Xcode command line tools version

## Workload Generation

**Random** means each lookup is an independent 7-card set; generation must not appear in timed region.

1. **Scale**: Generate 100K+ unique hand batches with fixed seed
   - Ensures cache pressure and prevents unrealistic cache hits

2. **Cache-warm phase**: Touch lookup tables and initial hands before timing

## Timer and Measurement

Use `std.time.nanoTimestamp()` (wraps `mach_absolute_time()` on macOS) at batch boundaries.

Measure in blocks of 65K+ batches (1M+ hands) to amortize timer overhead.

## Run Plan

1. **Multiple runs**: Execute 5 independent benchmark runs

2. **Cache clearing**: Between runs, clear caches with `sudo purge` and sleep 3 seconds

3. **Overhead isolation**: Run "dummy evaluator" that returns popcount to measure framework overhead

4. **Statistical analysis**: Report median time and coefficient of variation

## Metrics and Analysis

**Primary metric**: Median nanoseconds per hand across five runs, after overhead correction.

**Target performance**: 2-5ns per hand (as specified in DESIGN.md)

**Quality check**: Coefficient of variation < 5% (indicates consistent measurement)

## Correctness Guard

During warm-up:
- Compute rolling XOR checksum of all returned ranks
- Compare against known good reference
- Abort on mismatch
