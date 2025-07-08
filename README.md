# Zig Poker Evaluator & Analysis Toolkit

A comprehensive high-performance poker hand evaluator and analysis toolkit in Zig 0.14.0. Combines ultra-fast 7-card hand evaluation (~4.5ns per hand) with complete poker analysis capabilities including equity calculations, range analysis, and Monte Carlo simulations.

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
const poker = @import("poker");

// Single hand evaluation (7-card hand encoded as u64)
const hand: u64 = 0x1F00000000000; // Royal flush pattern
const rank = poker.evaluateHand(hand); // Lower rank = stronger hand

// SIMD batch evaluation (32 hands simultaneously)
var rng = std.Random.DefaultPrng.init(42).random();
const batch = poker.generateRandomHandBatch(32, &rng);
const results = poker.evaluateBatch(32, batch);

// Performance: ~4.5ns per hand, 224M+ hands/sec
```

### Poker Analysis (using integrated poker module)

```zig
const poker = @import("poker");

// Parse and evaluate a 7-card hand
const hand = poker.mustParseHand("AhAsKhQhJhTh2c");
const rank = poker.evaluateHand(hand); // Returns 1 (royal flush)
const category = poker.getHandCategory(rank); // .straight_flush

// Alternative: create cards individually
const ace_hearts = poker.makeCard(.hearts, .ace);
const ace_spades = poker.makeCard(.spades, .ace);
// ... combine into hand using bitwise OR
```

### Equity Calculation

```zig
const poker = @import("poker");

// Calculate preflop equity
const aa = poker.mustParseHand("AhAs");
const kk = poker.mustParseHand("KdKc");

var prng = std.Random.DefaultPrng.init(42);
const result = try poker.monteCarlo(aa, kk, 0, 100000, prng.random(), allocator);
// result.equity() ≈ 0.80 (80% equity for AA vs KK preflop)

// Multi-way equity with board
const qq = poker.mustParseHand("QhQs");
const board = poker.mustParseHand("AdKh7s");

const hands = [_]poker.Hand{ aa, kk, qq };
const results = try poker.multiway(&hands, board, 50000, prng.random(), allocator);
defer allocator.free(results);
// results[0].equity() ≈ 0.42 (AA equity in 3-way pot)

// Exact calculation (small boards only)
const exact_result = try poker.exact(aa, kk, board, allocator);
// Enumerates all possible outcomes for perfect accuracy
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

// Use predefined ranges
var button_range = try poker.CommonRanges.buttonOpen(allocator);
defer button_range.deinit();
// Contains: AA-22, AK-A2, KQ-K8, suited connectors, etc.
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
  Total hands:         3200000
  Framework overhead:  0.62 ns/hand
  Batch performance:   4.46 ns/hand
  Hands per second:    224,293,824
  Variation:           5.28% (high - consider stable environment)
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
- **`evaluator.zig`**: SIMD-optimized batch evaluation (~4.5ns per hand)
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
│   ├── evaluator.zig    # SIMD-optimized evaluation (~4.5ns)
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

## API Reference

### Core Types
- `Hand` - 64-bit bitfield representing cards
- `Suit` - Card suits (clubs=0, diamonds=1, hearts=2, spades=3)
- `Rank` - Card ranks (two=0 through ace=12)
- `HandRank` - Evaluation result (0-7461, lower = stronger)
- `HandCategory` - Categories from high_card to straight_flush
- `Range` - Poker range with hand combinations and probabilities

### Hand Creation & Parsing
```zig
// Parse hands and cards
const hand = poker.mustParseHand("AhKsQdJcTh9s8c");  // 7-card hand
const cards = poker.mustParseHand("AhKs");           // Any number of cards

// Create individual cards
const card = poker.makeCard(.hearts, .ace);          // Using enums
const card2 = poker.parseCard("Ks") catch unreachable; // Parse single card
```

### Hand Evaluation
```zig
// Single hand evaluation
const rank = poker.evaluateHand(hand);               // Returns 0-7461
const category = poker.getHandCategory(rank);        // e.g., .two_pair

// SIMD batch evaluation
const batch = poker.generateRandomHandBatch(32, &rng);
const results = poker.evaluateBatch(32, batch);      // 32 hands (optimal)
```

### Equity Calculations
```zig
// Monte Carlo simulation
const equity = try poker.monteCarlo(hero, villain, board, 100000, rng, allocator);

// Exact enumeration (for small boards)
const exact = try poker.exact(hero, villain, board, allocator);

// Multi-threaded calculation
const threaded = try poker.threaded(hero, villain, board, 100000, seed, allocator);

// Multi-way pots
const hands = [_]poker.Hand{ aa, kk, qq };
const results = try poker.multiway(&hands, board, 50000, rng, allocator);
```

### Range Operations
```zig
// Parse poker ranges
var range = try poker.parseRange("AA,KK,QQ,AKs", allocator);
defer range.deinit();

// Use predefined ranges
var tight = try poker.CommonRanges.tightOpen(allocator);
var loose = try poker.CommonRanges.looseCall(allocator);
var button = try poker.CommonRanges.buttonOpen(allocator);

// Generate specific combinations
const suited = try poker.generateSuitedCombinations(.ace, .king, allocator);
const offsuit = try poker.generateOffsuitCombinations(.ace, .king, allocator);
const pairs = try poker.generatePocketPair(.ace, allocator);
```
