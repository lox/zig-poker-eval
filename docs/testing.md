<!-- Generated: 2025-01-09 10:45:30 UTC -->

# Testing Documentation

## Overview

The Zig poker evaluator uses a comprehensive testing approach with 82 unit tests distributed across modules, integration tests, benchmarks, and verification tools. Tests are written inline with the source code using Zig's built-in testing framework.

### Test File Locations

- **Core Library Tests**: `src/*.zig` - Each module contains inline tests
- **Internal Tests**: `src/internal/*.zig` - Low-level implementation tests
- **Test Runner**: `src/tools/test_runner.zig` - Custom test runner with detailed output
- **Benchmarks**: `src/tools/benchmark.zig` - Performance benchmarking framework
- **Verification Tools**: `src/tools/verify_all_hands.zig` - Exhaustive correctness testing

## Test Types

### Unit Tests
Unit tests are embedded directly in source files using Zig's `test` blocks:

```zig
test "hand category conversion" {
    try testing.expect(getHandCategory(1) == .straight_flush);
    try testing.expect(getHandCategory(100) == .four_of_a_kind);
    try testing.expect(getHandCategory(7000) == .high_card);
}
```

Test distribution by module:
- `card.zig` (line 181-274): 8 tests - Card creation, formatting, parsing
- `evaluator.zig` (line 325-516): 10 tests - Hand evaluation, batch processing
- `hand.zig` (line 143-269): 7 tests - Hand parsing, combinations
- `range.zig` (line 404-506): 7 tests - Range notation and parsing
- `equity.zig` (line 719-779): 4 tests - Equity calculations
- `analysis.zig` (line 272-330): 5 tests - Board texture analysis
- `draws.zig` (line 544-680): 7 tests - Draw detection
- `poker.zig` (line 244-282): 3 tests - Public API
- `internal/slow_evaluator.zig` (line 281-447): 17 tests - Reference implementation
- `internal/notation.zig` (line 176-316): 6 tests - Notation parsing
- `internal/simulation.zig` (line 170-242): 4 tests - Monte Carlo helpers
- `internal/build_tables.zig` (line 39-90): 4 tests - Table generation

### Integration Tests
- **Exhaustive Verification**: `verify-all` tests all 133M possible 7-card hands
- **Cross-validation**: Tests verify fast evaluator against slow reference implementation

### Performance Benchmarks
The benchmark suite (`src/tools/benchmark.zig`) measures:
- Single hand evaluation performance (target: 2-5ns)
- Batch evaluation with SIMD (32 hands at once)
- Cache warming and overhead measurement
- Statistical analysis (coefficient of variation)

## Running Tests

### Basic Test Commands

```bash
# Run all unit tests (82 tests across all modules)
zig build test

# Run tests with detailed summary
zig build test --summary all

# Run specific module tests (via test runner)
zig test src/evaluator.zig

# Run performance benchmark
zig build bench -Doptimize=ReleaseFast

# Generate all 133M hands for verification
zig build gen-all

# Verify evaluator correctness
zig build verify-all
```

### Expected Test Output

Standard test run:
```
$ zig build test
✅ All 82 tests passed (0 skipped, 0 leaked)
```

Detailed test runner output (via custom runner):
```
Test                                               Status    Time (ms)
----------------------------------------------------------------------
card.card creation and format                      PASS          0.123
card.enum consistency                              PASS          0.045
evaluator.hand category conversion                 PASS          0.067
evaluator.flush pattern extraction                 PASS          0.089
...
----------------------------------------------------------------------
Total time: 45.678 ms
✅ All 82 tests passed (0 skipped, 0 leaked)
```

Benchmark output:
```
$ zig build bench -Doptimize=ReleaseFast
=== Poker Hand Evaluator Benchmark ===
Iterations: 100000
Hands per iteration: 32
Total hands evaluated: 3200000

Results:
  Batch evaluation: 2.3 ns/hand
  Single evaluation: 3.1 ns/hand
  SIMD speedup: 1.35x
  Throughput: 434M hands/second
  CV: 0.8%
```

## Test File Organization

### Build Configuration
The `build.zig` file configures test execution:
- Each module is tested individually with proper dependencies (line 153-243)
- Tests are added to the main test step for parallel execution
- Test artifacts use the same target and optimization settings

### Module Dependencies
Tests respect the module hierarchy:
1. **Level 1**: `card.zig` - No dependencies
2. **Level 2**: `evaluator.zig` - Depends on card
3. **Level 3**: `hand.zig`, `equity.zig` - Depend on card/evaluator
4. **Level 4**: `range.zig`, `analysis.zig`, `draws.zig` - Higher-level APIs
5. **Level 5**: `poker.zig` - Main API depends on all modules

### Custom Test Runner
The test runner (`src/tools/test_runner.zig`) provides:
- Formatted table output with test names, status, and timing
- Memory leak detection via Zig's testing allocator
- Total execution time tracking
- Exit code based on test results

## Build Targets Reference

| Target | Description | Command |
|--------|-------------|---------|
| `test` | Run all unit tests | `zig build test` |
| `bench` | Performance benchmark | `zig build bench -Doptimize=ReleaseFast` |
| `gen-all` | Generate all hands | `zig build gen-all` |
| `verify-all` | Verify correctness | `zig build verify-all` |
| `build-tables` | Regenerate lookup tables | `zig build build-tables -Doptimize=ReleaseFast` |

## Test Best Practices

1. **Inline Tests**: Tests are colocated with implementation for easy maintenance
2. **Fast Execution**: Unit tests run in milliseconds for rapid feedback
3. **Comprehensive Coverage**: Every public API has corresponding tests
4. **Edge Cases**: Special attention to boundary conditions (e.g., two trips making full house)
5. **Performance Validation**: Benchmarks ensure optimizations don't regress
