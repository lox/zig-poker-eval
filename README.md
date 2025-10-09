<!-- Generated: 2025-07-09 07:24:55 UTC -->

# Zig Poker Evaluator

High-performance 7-card poker hand evaluator achieving ~4.5ns per hand evaluation on Apple M1. Includes comprehensive analysis tools for equity calculations, range parsing, and Monte Carlo simulations.

## Installation

Requires Zig 0.15.1 or later.

Add zig-poker-eval as a dependency in your `build.zig.zon`:

```bash
zig fetch --save "git+https://github.com/lox/zig-poker-eval?ref=v2.0.0"
```

In your `build.zig`, add the `poker` module as a dependency to your program:

```zig
const poker = b.dependency("zig_poker_eval", .{
    .target = target,
    .optimize = optimize,
});

// the executable from your call to b.addExecutable(...)
exe.root_module.addImport("poker", poker.module("poker"));
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

## Development

Uses [Hermit](https://github.com/cashapp/hermit) for dependencies and [Task](https://taskfile.dev) for automation.

```bash
# Activate Hermit
source bin/activate-hermit

# Build
task build

# Run tests (82 tests across modules)
task test

# Benchmark performance
task bench:eval

# Calculate equity
task run -- equity "AhAs" "KdKc"
```

## Performance

Achieved on Apple M1:

- Single evaluation: ~4.5ns per hand
- Batch evaluation: 224M+ hands/second
- Memory usage: ~267KB for lookup tables (L2 cache resident)
