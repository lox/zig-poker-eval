<!-- Generated: 2025-07-09 07:24:55 UTC -->

# Zig Poker Evaluator

High-performance 7-card poker hand evaluator achieving ~4.5ns per hand evaluation on Apple M1. Includes comprehensive analysis tools for equity calculations, range parsing, and Monte Carlo simulations.

```bash
# Build and run
zig build run

# Run tests (82 tests across modules)
zig build test

# Benchmark performance
zig build bench -Doptimize=ReleaseFast
```

## Documentation

### Core Documentation

- **[Project Overview](docs/project-overview.md)** - What the project does, technology stack, platform support
- **[Architecture](docs/architecture.md)** - System design, component map, data flows, perfect hash implementation
- **[Build System](docs/build-system.md)** - Build commands, module structure, optimization flags
- **[Testing](docs/testing.md)** - Test organization, running tests, performance benchmarks
- **[Development](docs/development.md)** - Code style, patterns, workflows, adding features
- **[Deployment](docs/deployment.md)** - Library packaging, executable distribution, platform builds
- **[Files Catalog](docs/files.md)** - Complete file listing with descriptions and relationships

### Performance Documentation

- **[Design](docs/design.md)** - Implementation details, algorithms, and trade-offs
- **[Benchmarking](docs/benchmarking.md)** - Performance measurement methodology and results
- **[Profiling](docs/profiling.md)** - CPU profiling tools and optimization workflow
- **[Experiments](docs/experiments.md)** - Optimization experiments and lessons learned

## Core Features

- **Ultra-fast evaluation**: ~4.5ns per hand using CHD perfect hash tables and SIMD batch processing
- **SIMD optimization**: Batch evaluation of 32 hands simultaneously
- **Equity calculations**: Monte Carlo and exact enumeration
- **Range parsing**: Standard poker notation (AA, KK, AKs, etc.)
- **Board analysis**: Texture, draws, and hand strength assessment
- **Zero dependencies**: Pure Zig implementation

## Example Usage

```zig
const poker = @import("poker");

// Evaluate a hand
const hand = poker.parseHand("AsKsQsJsTs5h2d");
const rank = poker.evaluateHand(hand);  // Returns 1 (royal flush)

// Calculate equity
const aa = poker.parseHand("AhAs");
const kk = poker.parseHand("KdKc");
const result = try poker.monteCarlo(aa, kk, &.{}, 100000, rng, allocator);
// result.equity() ≈ 0.80 (AA wins ~80% vs KK)
```

## Performance

Achieved on Apple M1:

- Single evaluation: ~4.5ns per hand
- Batch evaluation: 224M+ hands/second
- Memory usage: ~267KB for lookup tables (L2 cache resident)
