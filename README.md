# Zig Poker Evaluator & Analysis Toolkit

A comprehensive high-performance poker hand evaluator and analysis toolkit in Zig 0.14.0. Combines ultra-fast 7-card hand evaluation (~8ns per hand) with complete poker analysis capabilities including equity calculations, range analysis, and Monte Carlo simulations.

## Setup

Uses [Hermit](https://github.com/cashapp/hermit) for dependency management:

```bash
. bin/activate-hermit  # Activates Zig environment
```

## Usage

```bash
# Show help and available commands
zig build run

# Run demo (hand evaluation + SIMD batching)
zig build run -- demo

# Parse poker ranges
zig build run -- range "AA,KK,AKs"

# Run comprehensive benchmark
zig build bench -Doptimize=ReleaseFast

# Quick benchmark with validation
zig build run -- bench --quick --validate

# Run all tests (60+ test cases)
zig build test
```

## Examples

### Hand Evaluation

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

### Equity Calculation

```zig
const equity = @import("equity.zig");
const poker = @import("poker.zig");

// Calculate preflop equity using simplified parsing
const aa = poker.mustParseHoleCards("AhAs");
const kk = poker.mustParseHoleCards("KdKc");

var prng = std.Random.DefaultPrng.init(42);
const result = try equity.monteCarlo(aa, kk, &.{}, 100000, prng.random(), allocator);
// result.equity() ≈ 0.80 (80% equity for AA vs KK preflop)

// Multi-way equity with postflop board
const qq = poker.mustParseHoleCards("QhQs");
const board_cards = poker.mustParseCards("AdKh7s");

var hands = [_][2]poker.Card{ aa, kk, qq };
const results = try equity.multiway(&hands, &board_cards, 50000, prng.random(), allocator);
defer allocator.free(results);
// results[0].equity() ≈ 0.42 (AA equity in 3-way pot)
```

### Range vs Range Equity

```zig
const ranges = @import("ranges.zig");

// Create ranges using standard poker notation
var hero_range = try ranges.parseRange("AA,KK,QQ,AKs,AQs", allocator);  // Tight range
defer hero_range.deinit();

var villain_range = try ranges.parseRange("TT,99,88,KQs,QJs", allocator); // Calling range
defer villain_range.deinit();

// Calculate range vs range equity
const result = try ranges.calculateRangeEquityMonteCarlo(
    &hero_range, &villain_range, &.{}, 10000, rng, allocator
);
// Hero: 49.5%, Villain: 50.5%
```

### Range Notation Syntax

```
AsAh       specific cards (ace of spades and ace of hearts)
AA         pocket pairs (any pair of aces)
AA, KK     comma-separated ranges (aces or kings)
JTs        suited combinations (jack-ten suited)
JTo        offsuit combinations (jack-ten offsuit)
JT         any combinations (jack-ten suited + offsuit)
AA, KK, AK mixed ranges (pocket pairs + unpaired hands)
```
## Performance

Benchmarked on an Apple Macbook Air M1.

```bash
zig build bench -Doptimize=ReleaseFast

🚀 Performance Benchmark
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Running benchmark with 100000 iterations...
  • Cache warmup enabled
  • Overhead measurement enabled
  • Multiple runs for statistical analysis

📊 Benchmark Results
  Total hands:         400000
  Framework overhead:  0.72 ns/hand
  Batch performance:   6.57 ns/hand
  Hands per second:    152,224,378
  Variation:           1.71% (low - reliable measurement)

🔄 Performance Comparison
  Batch (4x SIMD):     6.57 ns/hand (152,224,378 hands/sec)
  Single hand:         10.70 ns/hand (93,457,944 hands/sec)
  SIMD Speedup:        1.63x
```

### CLI Commands

```bash
# Range analysis
poker-eval range "AA,KK,AKs"              # 16 combinations
poker-eval range "AA-TT,AKs,AQs" --verbose

# Hand evaluation (coming soon)
poker-eval eval "AhAsKhQsJhThTc"

# Equity analysis (coming soon)
poker-eval equity "AhAs" "KdKc" --sims 100000

# Advanced benchmarking
poker-eval bench --iterations 1000000     # Custom iterations
poker-eval bench --validate --test        # With validation
poker-eval bench --test-hand 0x1F00       # Test specific hands
```

### Architecture

#### Evaluator Core (`src/evaluator/`)
- **`evaluator.zig`**: SIMD-optimized batch evaluation (2-7ns per hand)
- **`slow_evaluator.zig`**: Reference implementation for validation
- **`tables.zig`**: Pre-computed perfect hash lookup tables (120KB)
- **`mphf.zig`**: CHD perfect hash function implementation
- **`build_tables.zig`**: Table generation utilities

#### Poker Analysis (`src/poker/`)
- **`poker.zig`**: Core poker types, Hand data structures, and comprehensive tests
- **`equity.zig`**: Monte Carlo and exact equity calculation with multithreading support
- **`simulation.zig`**: Low-level simulation primitives and showdown evaluation
- **`ranges.zig`**: Hand range generation and parsing utilities with clean HandKey abstraction
- **`notation.zig`**: Poker notation parsing (AhKs, AKo, ranges)

#### User Interface (`src/cli/`)
- **`main.zig`**: Professional CLI with multiple commands and colored output
- **`ansi.zig`**: Terminal color and formatting utilities

#### Tools (`src/tools/`)
- **`benchmark.zig`**: Comprehensive benchmarking framework with statistical analysis

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
19 equity.monteCarlo + 604  /src/equity.zig:48     (73% of samples)
6  equity.monteCarlo + 488  /src/equity.zig:40     (23% of samples)
1  equity.monteCarlo + 776  /src/equity.zig:52     (4% of samples)
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
├── cli/
│   ├── main.zig         # Professional CLI interface
│   └── ansi.zig         # Terminal color utilities
├── evaluator/
│   ├── mod.zig          # Public evaluator API
│   ├── evaluator.zig    # SIMD-optimized evaluation (2-7ns)
│   ├── slow_evaluator.zig # Reference implementation
│   ├── tables.zig       # Perfect hash lookup tables (120KB)
│   ├── mphf.zig         # CHD perfect hash implementation
│   └── build_tables.zig # Table generation
├── poker/
│   ├── mod.zig          # Public poker analysis API
│   ├── poker.zig        # Core types and hand evaluation
│   ├── equity.zig       # Monte Carlo equity calculations
│   ├── ranges.zig       # Range parsing and analysis
│   ├── notation.zig     # Poker notation parsing
│   └── simulation.zig   # Showdown simulations
├── tools/
│   └── benchmark.zig    # Comprehensive benchmarking
└── test.zig             # Test orchestration (60+ tests)

docs/                    # Technical documentation
CLAUDE.md               # AI coding assistant instructions
build.zig               # Build configuration with modules
bin/                    # Hermit-managed Zig installation
```
