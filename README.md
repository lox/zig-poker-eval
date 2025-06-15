# Zig Poker Hand Evaluator

A high-performance 7-card Texas Hold'em hand evaluator written in Zig.

## Setup

Uses [Hermit](https://github.com/cashapp/hermit) for dependency management:

```bash
. bin/activate-hermit  # Activates Zig environment
```

## Usage

```bash
# Run demo (hand evaluation + equity examples)
zig build run -Doptimize=ReleaseFast

# Run benchmarks only
zig build bench -Doptimize=ReleaseFast

# Run tests
zig build test

# Run comprehensive performance profiling
zig build profile -Doptimize=ReleaseFast
```

## Example

```zig
const poker = @import("poker.zig");

// Create a 7-card hand (hole cards + community cards)
const hand = poker.createHand(&.{
    .{ .hearts, .ace },   // Hole card 1
    .{ .spades, .ace },   // Hole card 2
    .{ .hearts, .king },  // Flop
    .{ .hearts, .queen }, // Flop
    .{ .hearts, .jack },  // Flop
    .{ .hearts, .ten },   // Turn
    .{ .clubs, .two },    // River
});

const rank = hand.evaluate(); // .straight_flush (royal flush)
```

## Equity Calculation

```zig
const equity = @import("equity.zig");

// Calculate preflop equity between pocket aces and pocket kings
const aa = [_]poker.Card{
    poker.createCard(.hearts, .ace),
    poker.createCard(.spades, .ace),
};
const kk = [_]poker.Card{
    poker.createCard(.diamonds, .king),
    poker.createCard(.clubs, .king),
};

var prng = std.Random.DefaultPrng.init(42);
const result = try equity.equityMonteCarlo(aa, kk, &.{}, 100000, prng.random(), allocator);
// result.equity() ≈ 0.80 (80% equity for AA vs KK preflop)

// Multi-way equity with postflop board
const qq = [_]poker.Card{
    poker.createCard(.hearts, .queen),
    poker.createCard(.spades, .queen),
};
const board = [_]poker.Card{
    poker.createCard(.diamonds, .ace),   // Flop
    poker.createCard(.hearts, .king),    // Flop
    poker.createCard(.spades, .seven),   // Flop
};

var hands = [_][2]poker.Card{ aa, kk, qq };
const results = try equity.equityMultiWayMonteCarlo(&hands, &board, 50000, prng.random(), allocator);
// results[0].equity() ≈ 0.42 (AA equity in 3-way pot)
```

## Performance

Benchmarked on an Apple Macbook Air M1.

```bash
zig build bench -Doptimize=ReleaseFast
=== Zig 7-Card Texas Hold'em Evaluator ===
=== Benchmark  ===
Generating 10000 random hands...
Run 1: 52080000 ops, 19.00 ns/op
Run 2: 54750000 ops, 18.00 ns/op
Run 3: 55780000 ops, 17.00 ns/op

=== Performance Summary ===
18.00 ns/op (average across 3 runs)
55.6M evaluations/second
```

### Architecture

- **`poker.zig`**: Core evaluation logic, data structures, and comprehensive tests
- **`equity.zig`**: Monte Carlo and exact equity calculation for poker hands
- **`simulation.zig`**: Low-level simulation primitives and showdown evaluation
- **`ranges.zig`**: Hand range generation and parsing utilities
- **`benchmark.zig`**: Performance testing and random hand generation
- **`profiler.zig`**: Advanced profiling system for performance analysis
- **`main.zig`**: Demo showcasing hand evaluation and equity calculation
- **`bench_main.zig`**: Dedicated benchmark executable

## Profiling

Two profiling approaches are available for performance analysis:

### 1. External System Profiling (Recommended)
Uses macOS `sample` command for real execution context profiling (macOS only):

```bash
# Profile both benchmarks (debug mode for better symbol resolution)
./scripts/profile_bench.sh

# Profile specific components
./scripts/profile_bench.sh eval    # Hand evaluation only
./scripts/profile_bench.sh equity  # Equity calculation only
```

Both will generate a `profile_output.txt` file in the root directory.

**Sample Output:**
```
Top functions by sample count:
19 equity.equityMonteCarlo + 604  /src/equity.zig:48     (73% of samples)
6  equity.equityMonteCarlo + 488  /src/equity.zig:40     (23% of samples)
1  equity.equityMonteCarlo + 776  /src/equity.zig:52     (4% of samples)
```

### 2. Custom Micro-Benchmarking
For detailed component-level analysis and algorithm comparisons:

```bash
# Run comprehensive micro-benchmark analysis
zig build profile -Doptimize=ReleaseFast
```

**Sample Output:**
```
Function                       Calls   Total (ms)   Avg (ns)   Min (ns)   Max (ns)
--------------------------------------------------------------------------------
full_evaluation               100000        1.555       15.6          0       1000
rank_extraction               100000        1.634       16.3          0       1000
flush_detection               100000        1.609       16.1          0       2000
straight_detection            100000        1.575       15.8          0       1000
pair_counting                 100000        1.590       15.9          0       2000
```

For detailed optimization techniques and experimental results, see `EXPERIMENTS.md`.

## Project Structure

```
src/
├── main.zig         # Demo showcasing hand evaluation and equity
├── poker.zig        # Core poker types, evaluation logic, and tests
├── equity.zig       # Monte Carlo and exact equity calculation
├── simulation.zig   # Low-level simulation primitives
├── ranges.zig       # Hand range generation and parsing
├── benchmark.zig    # Performance testing utilities
├── bench_main.zig   # Dedicated benchmark executable
├── profiler.zig     # Custom micro-benchmarking system
└── profile_main.zig # Profiling executable entry point

scripts/
├── profile_bench.sh   # External system profiling (recommended)

AGENT.md          # AI coding assistant instructions
CLAUDE.md         # AI coding assistant instructions
EXPERIMENTS.md    # Detailed optimization research and results
build.zig         # Build configuration
bin/              # Hermit-managed Zig installation
```
