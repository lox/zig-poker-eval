# Zig Poker Evaluator

High-performance 7-card poker hand evaluator achieving ~3.3ns per hand evaluation on Apple M1.

## Features

- **Ultra-fast evaluation**: ~3.3ns per hand using CHD perfect hash tables and SIMD batch processing
- **SIMD optimization**: Batch evaluation of 32 hands simultaneously
- **Equity calculations**: Monte Carlo and exact enumeration
- **Range parsing**: Standard poker notation (AA, KK, AKs, etc.)
- **Zero dependencies**: Pure Zig implementation

## Installation

Requires Zig 0.15.1 or later.

```bash
zig fetch --save "git+https://github.com/lox/zig-poker-eval?ref=v2.0.0"
```

In your `build.zig`:

```zig
const poker = b.dependency("zig_poker_eval", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("poker", poker.module("poker"));
```

## Usage

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

Uses [Hermit](https://github.com/cashapp/hermit) and [Task](https://taskfile.dev).

```bash
# Activate environment
source bin/activate-hermit

# Build
task build

# Test
task test

# Benchmark
task bench:eval

# Calculate equity
task run -- equity "AhAs" "KdKc"
```

## Performance

Apple M1 results:

- Single evaluation: ~3.3ns per hand
- Batch evaluation: 306M hands/second
- Speedup: 3.65× from baseline (11.95ns → 3.27ns)
- Memory: ~395KB lookup tables (267KB CHD + 128KB flush patterns)

## Documentation

- [docs/design.md](docs/design.md) - Architecture, algorithms, implementation
- [docs/performance.md](docs/performance.md) - Benchmarking and profiling
- [docs/experiments.md](docs/experiments.md) - Optimization experiments log

## License

MIT
