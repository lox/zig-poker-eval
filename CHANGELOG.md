# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- **Unified equity result types**: `EquityResult`, `DetailedEquityResult`, and `DetailedExactResult` have been merged into a single `EquityResult` type with optional category tracking fields
- Internal code consolidation reduced `src/equity.zig` from 1247 to 1103 lines (11.5% reduction)
- `detailedMonteCarlo` now uses SIMD batching for improved performance

### Migration Guide

The public API remains backward compatible via type aliases. Existing code will continue to work without changes.

#### Result Types

**Before:**
```zig
const basic: poker.EquityResult = try poker.monteCarlo(...);
const detailed: poker.DetailedEquityResult = try poker.detailedMonteCarlo(...);
const exact_detailed: poker.DetailedExactResult = try poker.exactDetailed(...);
```

**After (all return unified `EquityResult`):**
```zig
const basic: poker.EquityResult = try poker.monteCarlo(...);
// basic.hand1_categories == null
// basic.confidenceInterval() == null

const detailed: poker.EquityResult = try poker.detailedMonteCarlo(...);
// detailed.hand1_categories.?.pair - access category data
// detailed.confidenceInterval().? - access confidence interval

const exact_detailed: poker.EquityResult = try poker.exactDetailed(...);
// exact_detailed.hand1_categories.?.flush - access category data
// exact_detailed.confidenceInterval() == null (exact has no CI)
```

**Key Changes:**
- `hand1_categories` and `hand2_categories` are now `?HandCategories` (optional) instead of required fields
- `confidenceInterval()` returns `?struct { lower: f64, upper: f64 }` instead of a required struct
- Access category data with `.?` unwrap operator when using detailed variants
- Type aliases `DetailedEquityResult` and `DetailedExactResult` point to `EquityResult` for backward compatibility

#### Removed Functions

The following trivial wrapper functions have been removed:

**`exactHeadToHead`:**
```zig
// Before
const result = try poker.exactHeadToHead(hero, villain, allocator);

// After - use exact() with empty board
const result = try poker.exact(hero, villain, &.{}, allocator);
```

### Performance

No regression in performance. Baseline measurements on Apple M1:

**Hand Evaluation:**
- Single: ~3.3ns per hand (261M hands/sec)
- Showdown: 7.88ns per eval (3.87x speedup vs context path)

**Equity Calculations:**
- Preflop: 14.03M simulations/sec
- Flop: 40.98M simulations/sec
- Turn: 64.52M simulations/sec
- River: 142.86M simulations/sec

## [2.0.0] - Previous Release

See git history for earlier changes.
