# Profiling Scripts

Scripts for performance analysis and profiling of the poker hand evaluator.

## Scripts

### `profile.sh`
**Unified profiling script** - High-frequency sampling with configurable parameters.

```bash
./scripts/profile.sh [iterations] [duration] [output_file]
```

**Examples**:
```bash
# Quick profile (default: 20M iterations, 15s sampling)
./scripts/profile.sh

# Custom parameters
./scripts/profile.sh 5000000 10 quick_profile.txt

# Long detailed profile
./scripts/profile.sh 50000000 30 detailed_profile.txt
```

**Features**:
- Builds benchmark with ReleaseFast + debug symbols
- Configurable iteration count (default: 20M)
- Configurable sampling duration (default: 15s)
- Samples every 1ms for high resolution
- Outputs detailed function-level breakdown
- Error handling for failed sampling

**Use this for identifying performance bottlenecks.**


## Requirements

- **macOS only** (uses `sample` command)
- Benchmark must be built with debug symbols
- For other platforms: Use `perf` (Linux) or VTune (Windows)

## Usage Notes

- Scripts automatically handle argument passing to benchmark
- Use `simple_profile.sh` for most profiling needs
- High iteration counts (5M-20M) provide better sampling data
- Profile output shows line-level hotspots in source code

## Example Output

```
=== PROFILING RESULTS ===
Total samples: 3886

Function breakdown:
        evaluator.evaluate_hand         3815 (98.2%)
        main                           71   (1.8%)
```

This clearly identifies the bottleneck functions for optimization.

## LLVM IR Analysis

For deep compiler optimization analysis, generate LLVM IR to inspect vectorization behavior:

### Generate LLVM IR

```bash
# Generate LLVM IR for the benchmark executable
zig build-exe src/bench.zig -O ReleaseFast -femit-llvm-ir=bench.ll

# View the generated IR file
ls -la bench.ll
```

### Analyzing Vectorization

Look for SIMD vectorization patterns:

```bash
# Search for vector operations
grep -C 5 "vector\|simd\|<4 x" bench.ll

# Find specific function implementations
grep -A 100 "define.*evaluate_batch_4" bench.ll

# Look for control flow that breaks vectorization
grep -A 20 "extractelement\|br i1" bench.ll
```

### Key Patterns to Identify

**Good SIMD code**:
- `<4 x i64>` vector types maintained throughout
- Vector operations like `lshr <4 x i64>`, `and <4 x i16>`
- Minimal `extractelement` calls

**Bad SIMD code**:
- Immediate `extractelement` calls breaking vectors apart
- Sequential processing with `br i1` branches per element
- Loop unrolling instead of vectorization

### Example Analysis

The LLVM IR revealed our main issue:
```llvm
define internal fastcc <4 x i16> @evaluator.evaluate_batch_4(<4 x i64> %0) {
  %1 = extractelement <4 x i64> %0, i64 0  ; ❌ Breaks apart vector
  %2 = extractelement <4 x i64> %0, i64 1  ; ❌ Individual processing
  %3 = extractelement <4 x i64> %0, i64 2  ; ❌ Defeats SIMD
  %4 = extractelement <4 x i64> %0, i64 3  ; ❌ Control flow complexity
```

This shows the compiler is processing hands individually due to flush detection branches, rather than maintaining SIMD parallelism.

### Compiler Optimization Flags

For additional vectorization insights:

```bash
# Enable vectorization remarks (if using Clang backend)
zig build-exe src/bench.zig -O ReleaseFast -fvectorize -Rpass-analysis=loop-vectorize

# Note: Zig may not expose all Clang vectorization flags directly
```

**Use LLVM IR analysis to identify why SIMD optimizations fail and validate fixes.**
