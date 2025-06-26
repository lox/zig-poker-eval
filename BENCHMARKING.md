# Benchmarking Methodology

## 1. Hardware and OS Setup

1. Document platform: CPU model, macOS version, memory configuration
2. Optimize for consistent performance:
   - Set high performance mode: `sudo pmset -a highperf 1`
   - Check thermal state: `pmset -g thermlog`
   - Close unnecessary applications
3. Minimize system interference:
   - Run benchmark with elevated priority: `sudo nice -20`
   - Consider disabling Spotlight indexing temporarily

## 2. Build Discipline

1. Build in release mode:
   ```
   zig build bench -Doptimize=ReleaseFast -Dcpu=native
   ```
2. Record build environment:
   - Zig version and build flags
   - macOS version and Xcode command line tools version

## 3. Workload Generation

**Random** means each lookup is an independent 7-card set; generation must not appear in timed region.

1. **Scale**: Generate 100K+ unique hand batches with fixed seed
   - Ensures cache pressure and prevents unrealistic cache hits
2. **Cache-warm phase**: Touch lookup tables and initial hands before timing

## 4. Timer and Measurement

- Use `std.time.nanoTimestamp()` (wraps `mach_absolute_time()` on macOS) at batch boundaries.
- Measure in blocks of 65K+ batches (1M+ hands) to amortize timer overhead.

## 5. Run Plan

1. **Multiple runs**: Execute 5 independent benchmark runs
2. **Cache clearing**: Between runs, clear caches with `sudo purge` and sleep 3 seconds
3. **Overhead isolation**: Run "dummy evaluator" that returns popcount to measure framework overhead
4. **Statistical analysis**: Report median time and coefficient of variation

## 6. Metrics and Analysis

- **Primary metric**: Median nanoseconds per hand across five runs, after overhead correction.
- **Target performance**: 2-5ns per hand (as specified in DESIGN.md)
- **Quality check**: Coefficient of variation < 5% (indicates consistent measurement)

## 7. Correctness Guard and Validation

During warm-up and validation:

1. Compute rolling XOR checksum of all returned ranks for each batch.
2. Compare against a known good reference (from slow evaluator or golden file).
3. Abort on mismatch.
4. For SIMD evaluators, validate that batch results match scalar evaluation for the same hands.
5. For BBHash, check:
   - Level 2 should have ≤ 5 remaining patterns (with γ=2.0)
   - If you see >50 "patterns placed without hash (fallback)", your table sizing is wrong
   - Royal flush pattern (0x1F00) must return rank 0, not a fallback rank
6. **Memory footprint assertions:**
   - Use `smaps` or equivalent to confirm RSS ≈ 280 KB plus code

### 7.1 Validation Technique Used in bench.zig

- During benchmarking, a rolling XOR checksum of all returned hand ranks is computed for each batch.
- This checksum is compared against a known good reference (from the slow evaluator or a golden file).
- If a mismatch is detected, the benchmark aborts, ensuring that performance numbers are only reported for correct implementations.
- For SIMD evaluators, batch results are also compared against scalar evaluation for the same hands to ensure bitwise correctness.
- This technique is robust against silent errors and ensures that performance measurements are not taken on broken or misaligned evaluators.
- See DESIGN.md for further details on validation methodology.

**⚠️ BBHash Table Sizing Note:**
When implementing BBHash table sizing, the load factor γ determines how many **more** slots you need than keys:
```zig
// CORRECT: Table size = number_of_keys * gamma
level_size = @as(u32, @intFromFloat(@as(f64, @floatFromInt(patterns.len)) * gamma));

// WRONG: This makes tables too small and causes fallback patterns
level_size = @as(u32, @intFromFloat(@as(f64, @floatFromInt(patterns.len)) / gamma));
```
With γ=2.0, you need **double** the number of slots as keys for the hash to work efficiently. Dividing by γ instead of multiplying creates undersized tables that force many patterns into expensive fallback paths.

> **Callout:** Our hand rank convention is 0 = Royal Flush (best), 7461 = worst high card. This is reversed from most open-source evaluators (e.g., 2+2, poker-eval, OMPEval); adjust comparisons when cross-checking.
