# Better Benchmark Tool

## Goals

- **Reduce code**: ~35% reduction through unified runner and suite structure
- **Improve UX**: Single tool, automatic comparison, clear output
- **Integrate regression detection**: Built-in baseline comparison with configurable thresholds

## Stability Improvements

Based on gpt-5-pro analysis, implemented pragmatic code-level improvements for benchmark stability:

### Priority 1: Interquartile Mean (IQM) ✅
- **Impact**: 30-50% reduction in CV
- **Implementation**: `calculateStatistics()` now uses IQM instead of simple mean
- Drops top 25% and bottom 25% of samples, uses mean of middle 50%
- Eliminates outliers from OS scheduler interrupts and measurement noise
- Calculates stddev and CV from IQM samples only

### Priority 2: std.time.Timer ✅
- **Impact**: Better timing precision, lower overhead
- **Implementation**: Replaced `std.time.nanoTimestamp()` with `std.time.Timer`
- Applied to all benchmark functions: `timeScalarShowdown`, `timeBatchedShowdown`, `benchEvalBatch`
- Platform-specific optimized timing with better resolution

### Priority 3: Median as Primary Metric ✅
- **Impact**: More robust to outliers than mean
- **Implementation**: Already using median in display and storage
- Display shows `stats.median`, storage saves `stats.median`
- More stable metric, less affected by skew

### Priority 4: Increased Sample Sizes ✅
- **Impact**: Better statistical confidence
- **Warmup runs**: 3 (up from 1)
- **Normal benchmarks**: 10 measurement runs (up from 5)
- **Baseline mode**: 100 measurement runs (up from 50)
- **Iteration counts**: 100,000 per run for eval and showdown benchmarks
- CV threshold: 5% for both warnings and baseline rejection

### Measurement Parameters
- **Warmup**: 3 runs to stabilize CPU frequency and cache state
- **Normal mode**: 10 runs with IQM statistics
- **Baseline mode**: 100 runs for maximum stability
- **Iteration counts with repeats** (repeat same dataset N times for longer measurements):
  - eval batch: 100,000 iterations × 32 hands = 3.2M evaluations per run
  - showdown context: 100,000 cases × 10 repeats = 1M evaluations (~30s/run)
  - showdown batch: 100,000 cases × 10 repeats = 1M evaluations (~7s/run)
  - equity monte_carlo: 1,000 simulations × 10K MC = 10M evaluations per run
  - equity exact_turn: 100 iterations × 50 repeats × 44 rivers = 220K enumerations (~25s/run)
  - range equity_monte_carlo: 10 iterations × 10 repeats × 1K MC (~125s/run)
- **Key insight**: Repeat measurements instead of increasing dataset size to maintain cache characteristics

### Future Considerations (Not Yet Implemented)

**Priority 5: Adaptive Sampling**
- Keep running measurements until CV drops below target threshold
- Cap at max_runs to prevent infinite loops
- Guarantees stability rather than hoping N runs is enough
- Good for CI environments where stability is critical
- Medium complexity: requires refactoring BenchmarkRunner.run()

## Architecture

### Build Mode-Specific Baseline Files

Files named by build mode (e.g., `benchmark-baseline-release-fast.json`):
- Debug builds → `benchmark-baseline-debug.json` (gitignored)
- ReleaseFast → `benchmark-baseline-release-fast.json` (committed on main)
- ReleaseSafe → `benchmark-baseline-release-safe.json` (gitignored)
- ReleaseSmall → `benchmark-baseline-release-small.json` (gitignored)

Auto-selects correct file based on `builtin.mode`, prevents cross-mode comparisons.

```json
{
  "version": "1.0",
  "commit": "v2.9.0-0-g094a04799452",
  "timestamp": "2024-01-15T10:30:00Z",
  "system": {
    "hostname": "macbook-pro.local",
    "cpu": "Apple M1 Pro",
    "arch": "aarch64",
    "os": "darwin",
    "build_mode": "ReleaseFast"
  },
  "suites": {
    "eval": {
      "batch_evaluation": {
        "unit": "ns/hand",
        "value": 3.01,
        "runs": 100,
        "cv": 0.023
      }
    },
    "showdown": {
      "context_path": {
        "unit": "ns/eval",
        "value": 27.90,
        "runs": 100,
        "cv": 0.018
      },
      "batched": {
        "unit": "ns/eval",
        "value": 12.45,
        "runs": 100,
        "cv": 0.021
      }
    }
  }
}
```

- Groups benchmarks by suite (eval, showdown, equity, etc.)
- Stores system metadata (hostname, CPU, arch, OS) for cross-platform safety
- Records commit and timestamp for context
- Simple, predictable workflow

### Suite Structure

```zig
BenchmarkSuite {
    name: "eval",
    benchmarks: [
        { name: "batch_evaluation", unit: "ns/hand", run_fn: ... },
        { name: "single_evaluation", unit: "ns/hand", run_fn: ... },
    ]
}
```

Each benchmark function returns a single number. Runner handles timing, warmup, statistics.

### CLI Commands

- `poker-eval bench` - Run all suites, compare with baseline, fail on regression
- `poker-eval bench --filter eval` - Run specific suite
- `poker-eval bench --baseline` - Save new baseline (100 runs, requires CV < 5%)
- `poker-eval bench --threshold 10.0` - Use custom regression threshold (default: 5%)

## Implementation Status

### Core Framework ✅ COMPLETE

- ✅ Define `BenchmarkSuite` and `Benchmark` structs
- ✅ Implement `BenchmarkRunner` with warmup, N runs, statistics (IQM, median, stddev, CV)
- ✅ Add `Statistics` struct with median, CV, min/max
- ✅ Implement progress indication (suite N of M)
- ✅ Add system info detection (hostname, CPU, arch, OS, build_mode)
- ✅ Add git commit detection (`git describe --tags --dirty --always --abbrev=12 --long`)

### Baseline Management ✅ COMPLETE

- ✅ Define baseline JSON schema with suites/benchmarks
- ✅ Implement `saveBaseline()` to write baseline file
- ✅ Implement `loadBaseline()` to read baseline file
- ✅ Add version field for future schema changes
- ✅ Include commit, timestamp, and system metadata in baseline
- ✅ Mode-specific baseline files (auto-selects by build mode)

### Comparison Engine ✅ COMPLETE

- ✅ Implement `compareResults()` function for two results
- ✅ Add threshold checking (configurable, default 5%)
- ✅ Generate comparison output (improvements/regressions with percentage change)
- ✅ Return appropriate exit codes (0 = pass, 1 = regression)
- ✅ Warn on system mismatch between baseline and current
- ✅ Hard error on build mode mismatch

### CLI Integration ✅ COMPLETE

- ✅ Add `bench` subcommand to main CLI
- ✅ Implement suite filtering with `--filter`
- ✅ Add `--baseline` flag to save baseline (with CV validation)
- ✅ Add `--threshold` flag for custom regression threshold
- ✅ Exit 1 on regression detection

### Port Existing Benchmarks ✅ COMPLETE

- ✅ Port hand evaluation benchmark (batch_evaluation)
- ✅ Port showdown benchmarks (context_path, batched)
- ✅ Port equity benchmarks (monte_carlo, exact_turn)
- ✅ Port range equity benchmark (equity_monte_carlo)
- ✅ All benchmarks integrated into unified suite structure
- ℹ️  No "batch sizes" benchmark found in codebase (likely doesn't exist)

### Testing & Validation ✅ COMPLETE

- ✅ Test benchmark runner with all 7 benchmarks
- ✅ Validate JSON serialization/deserialization
- ✅ Test comparison logic with build mode validation
- ✅ Verify exit codes on regression detection
- ✅ Test filtering by suite
- ✅ Validate system info detection
- ✅ End-to-end stability testing: 5 consecutive runs with 100% pass rate
- ✅ Iterative baseline refinement with repeated measurements
- ✅ Achieved final CVs: batch_evaluation (0.09%), context_path (0.14%), batched (0.30%), monte_carlo (0.51%), exact_turn (0.69%), equity_monte_carlo (0.11%)
- ⏸️  Cross-build-mode testing deferred (not critical for initial release)

### Documentation & CI ⏳ TODO

- [ ] Update `docs/performance.md` with new CLI usage
- [ ] Remove `scripts/compare_benchmark.sh` (if exists)
- [ ] Update CI workflow to use `poker-eval bench`
- [ ] Add baseline commit step for main branch merges
- [ ] Document benchmark stability improvements in README/changelog

## Success Criteria

### Core Functionality ✅ COMPLETE
- ✅ Unified benchmark framework (985 lines in benchmark.zig)
- ✅ No external dependencies (bash/jq/awk eliminated)
- ✅ Single tool handles run, compare, save baseline
- ✅ Automatic regression detection with exit codes
- ✅ Clear, consistent output format with Unicode symbols and throughput display
- ✅ Build mode-specific baselines with auto-selection
- ✅ Robust statistics (IQM, median, improved timing)
- ✅ **Exceptional measurement stability**: All benchmarks CV < 1% (100% pass rate)
- ✅ **4 benchmark suites**: eval, showdown, equity, range
- ✅ **7 total benchmarks**: batch_evaluation, context_path, batched, monte_carlo, exact_turn, equity_monte_carlo

### Remaining Work (Non-Critical)
- ⏳ Documentation updates (docs/performance.md, README/changelog)
- ⏳ CI workflow integration
- ⏳ Cleanup old scripts (scripts/compare_benchmark.sh if exists)
