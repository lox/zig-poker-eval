# Zig Poker Evaluator & Analysis Toolkit

A comprehensive high-performance poker hand evaluator and analysis toolkit in Zig 0.14.0. Combines ultra-fast 7-card hand evaluation (~7ns per hand) with complete poker analysis capabilities including equity calculations, range analysis, and Monte Carlo simulations.

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

# Parse poker ranges (currently working)
zig build run -- range "AA,KK,AKs"

# Run comprehensive benchmark
zig build bench -Doptimize=ReleaseFast

# Quick benchmark with validation
zig build run -- bench --quick --validate

# Run all tests (60+ test cases)
zig build test
```

## Examples

### High-Performance Hand Evaluation

```zig
const evaluator = @import("evaluator");

// Single hand evaluation (7-card hand encoded as u64)
const hand: u64 = 0x1F00000000000; // Royal flush pattern
const rank = evaluator.evaluateHand(hand); // Lower rank = stronger hand

// SIMD batch evaluation (4 hands simultaneously) 
var rng = std.Random.DefaultPrng.init(42).random();
const batch = evaluator.generateRandomHandBatch(&rng);
const results = evaluator.evaluateBatch4(batch); // 1.5x speedup

// Performance: ~7ns per hand, 140M+ hands/sec
```

### Poker Analysis (using integrated poker module)

```zig
const poker = @import("poker");

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
const poker = @import("poker");

// Calculate preflop equity using simplified parsing
const aa = poker.mustParseHoleCards("AhAs");
const kk = poker.mustParseHoleCards("KdKc");

var prng = std.Random.DefaultPrng.init(42);
const result = try poker.monteCarlo(aa, kk, &.{}, 100000, prng.random(), allocator);
// result.equity() ≈ 0.80 (80% equity for AA vs KK preflop)

// Multi-way equity with postflop board
const qq = poker.mustParseHoleCards("QhQs");
const board_cards = poker.mustParseCards("AdKh7s");

var hands = [_][2]poker.Card{ aa, kk, qq };
const results = try poker.evaluateShowdown(&hands, &board_cards, 50000, prng.random(), allocator);
defer allocator.free(results);
// results[0].equity() ≈ 0.42 (AA equity in 3-way pot)
```

### Range Analysis

```zig
const poker = @import("poker");

// Create ranges using standard poker notation
var hero_range = try poker.parseRange("AA,KK,QQ,AKs,AQs", allocator);  // Tight range
defer hero_range.deinit();

var villain_range = try poker.parseRange("TT,99,88,KQs,QJs", allocator); // Calling range
defer villain_range.deinit();

// Analyze range combinations
std.debug.print("Hero range: {} combinations\n", .{hero_range.handCount()});
std.debug.print("Villain range: {} combinations\n", .{villain_range.handCount()});

// Range vs range equity calculation (via poker module)
// Full implementation available through integrated analysis tools
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
# Range analysis (currently working)
zig build run -- range "AA,KK,AKs"              # 16 combinations
zig build run -- range "AA-TT,AKs,AQs" --verbose

# Hand evaluation (coming soon)
zig build run -- eval "AhAsKhQsJhThTc"

# Equity analysis (coming soon)  
zig build run -- equity "AhAs" "KdKc" --sims 100000

# Advanced benchmarking (fully functional)
zig build run -- bench --iterations 1000000     # Custom iterations
zig build run -- bench --validate --test        # With validation
zig build run -- bench --test-hand 0x1F00       # Test specific hands
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

## Benchmarking & Profiling

The unified CLI provides comprehensive benchmarking with statistical analysis:

### Built-in Benchmarking
```bash
# Full statistical benchmark with multiple runs
zig build bench -Doptimize=ReleaseFast

# Quick benchmark for development
zig build run -- bench --quick

# Advanced options
zig build run -- bench --iterations 1000000 --validate --test
zig build run -- bench --test-hand 0x1F00 --single-run
```

### Performance Analysis
```bash
# Test specific hand patterns
zig build run -- bench --test-hand 0x1F00      # Royal flush
zig build run -- bench --test-hand 0x123456    # Random hand

# Validate against reference implementation  
zig build run -- bench --validate              # Tests 16K hands

# Comprehensive testing
zig build run -- bench --test --validate       # Full evaluator test
```

The benchmark framework provides:
- **Statistical Analysis**: Multiple runs, median, coefficient of variation
- **Cache Warmup**: Proper cache preparation for accurate measurements  
- **Overhead Measurement**: Framework overhead calculation and subtraction
- **SIMD Comparison**: Single vs batch performance analysis
- **Correctness Validation**: 100% accuracy verification against reference

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
