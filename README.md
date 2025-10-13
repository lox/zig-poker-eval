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
zig fetch --save "git+https://github.com/lox/zig-poker-eval?ref=v3.6.1"
```

In your `build.zig`:

```zig
const poker = b.dependency("zig_poker_eval", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("poker", poker.module("poker"));
```

## Quick Start

```zig
const poker = @import("poker");

// Evaluate a hand
const hand = poker.parseHand("AsKsQsJsTs5h2d");
const rank = poker.evaluateHand(hand);  // Returns 1 (royal flush)

// Calculate equity - Monte Carlo simulation
const aa = poker.parseHand("AhAs");
const kk = poker.parseHand("KdKc");
const result = try poker.monteCarlo(aa, kk, &.{}, 100000, rng, allocator);
// result.equity() ≈ 0.82 (AA wins ~82% vs KK)
```

## Examples

### Hand Evaluation

```zig
const poker = @import("poker");

// Parse and evaluate a hand
const hand = poker.parseHand("AsKsQsJsTs5h2d");
const rank = poker.evaluateHand(hand);
const category = poker.getHandCategory(rank);
// category == .straight_flush (royal flush is a straight flush)

// Batch evaluation (SIMD optimized)
const hands = [_]u64{
    poker.parseHand("AsKsQsJsTs5h2d"),
    poker.parseHand("AhAdAcKhKd2s3s"),
    poker.parseHand("7c8c9cTcJcQdKd"),
};
const ranks = poker.evaluateBatch(3, hands);
```

### Basic Equity Calculation

```zig
var prng = std.Random.DefaultPrng.init(42);
const rng = prng.random();

const aa = poker.parseHand("AhAs");
const kk = poker.parseHand("KdKc");

// Monte Carlo simulation (fast, approximate)
const result = try poker.monteCarlo(aa, kk, &.{}, 100000, rng, allocator);
std.debug.print("AA vs KK: {d:.1}%\n", .{result.equity() * 100});
// Output: AA vs KK: 82.1%

// With board cards
const flop = [_]u64{
    poker.parseCard("Qh"),
    poker.parseCard("Jd"),
    poker.parseCard("Tc"),
};
const result_flop = try poker.monteCarlo(aa, kk, &flop, 50000, rng, allocator);
```

### Detailed Equity with Hand Categories

```zig
// Track how often each hand makes pairs, trips, flushes, etc.
const detailed = try poker.monteCarloWithCategories(aa, kk, &.{}, 100000, rng, allocator);

const hero_cats = detailed.hand1_categories.?;
std.debug.print("Hero makes:\n", .{});
std.debug.print("  Pair: {d:.1}%\n", .{hero_cats.percentage(hero_cats.pair)});
std.debug.print("  Two pair: {d:.1}%\n", .{hero_cats.percentage(hero_cats.two_pair)});
std.debug.print("  Set: {d:.1}%\n", .{hero_cats.percentage(hero_cats.three_of_a_kind)});
std.debug.print("  Flush: {d:.1}%\n", .{hero_cats.percentage(hero_cats.flush)});

// Get confidence interval
const ci = detailed.confidenceInterval().?;
std.debug.print("95% CI: [{d:.1}%, {d:.1}%]\n", .{ci.lower * 100, ci.upper * 100});
```

### Exact Equity (Deterministic)

```zig
// Enumerates all possible board runouts - slower but exact
const exact_result = try poker.exact(aa, kk, &.{}, allocator);
std.debug.print("Exact equity: {d:.4}\n", .{exact_result.equity()});
// Output: Exact equity: 0.8217 (no sampling variance)

// Fast on the turn (only 44 river cards)
const turn = [_]u64{
    poker.parseCard("Qh"),
    poker.parseCard("Jd"),
    poker.parseCard("Tc"),
    poker.parseCard("2s"),
};
const turn_exact = try poker.exact(aa, kk, &turn, allocator);
std.debug.print("Turn exact: {d:.2}%\n", .{turn_exact.equity() * 100});
```

### Range vs Range Equity

```zig
// Parse range notation
var hero_range = try poker.parseRange("AA,KK,AKs", allocator);
defer hero_range.deinit();

var villain_range = try poker.parseRange("22+,A2s+,K9s+,QTs+", allocator);
defer villain_range.deinit();

// Calculate range vs range equity
const range_result = try poker.calculateRangeEquityMonteCarlo(
    &hero_range,
    &villain_range,
    &.{},
    50000,
    rng,
    allocator,
);
defer range_result.deinit(allocator);

std.debug.print("Range equity: {d:.1}%\n", .{range_result.hero_equity * 100});
std.debug.print("Combinations evaluated: {}\n", .{range_result.total_combos});
```

### Fast Preflop Lookups

```zig
// Use pre-computed heads-up tables for instant results
const equity_table = poker.HeadsUpEquity{};

// Get exact equity for any starting hand matchup
const aa_vs_kk = equity_table.lookup(
    poker.StartingHand.fromString("AA").?,
    poker.StartingHand.fromString("KK").?,
);
std.debug.print("AA vs KK (instant): {d:.2}%\n", .{aa_vs_kk * 100});
// Output: AA vs KK (instant): 82.17%

// Check equity vs random for any hand
const ak_equity = poker.PREFLOP_VS_RANDOM[poker.StartingHand.fromString("AKs").?.index()];
std.debug.print("AKs vs random: {d:.1}%\n", .{ak_equity * 100});
```

### Multi-way Equity (Tournaments)

```zig
// 3+ player equity calculations
const hands = [_]u64{
    poker.parseHand("AhAs"),  // Hero
    poker.parseHand("KdKc"),  // Villain 1
    poker.parseHand("QhQs"),  // Villain 2
};

const multiway_results = try poker.multiway(&hands, &.{}, 50000, rng, allocator);
defer allocator.free(multiway_results);

for (multiway_results, 0..) |result, i| {
    std.debug.print("Player {}: {d:.1}%\n", .{ i + 1, result.equity() * 100 });
}
// Output:
// Player 1: 49.3%  (AA)
// Player 2: 28.1%  (KK)
// Player 3: 22.6%  (QQ)
```

### Multi-way Showdown Helper

```zig
// Compute the winning seats in one pass
const board = poker.parseHand("AsKsQsJs2h");
const ctx = poker.initBoardContext(board);
const seats = [_]u64{
    poker.parseHand("Th3c"), // Royal flush
    poker.parseHand("AhAd"), // Trips
    poker.parseHand("KcQc"), // Two pair
};

const showdown = poker.evaluateShowdownMultiway(&ctx, &seats);
std.debug.print("Best rank: {}\n", .{showdown.best_rank});
std.debug.print("Winner mask: 0b{b:0>3}\n", .{showdown.winner_mask});
std.debug.print("Tie count: {}\n", .{showdown.tie_count});
// Winner mask 0b001 → seat 0 wins outright
```

### Per-sample Equity Weights

```zig
// Normalized equity shares (1/tie_count for winners)
var equities = [_]f64{ 0, 0, 0 };
const weights = poker.evaluateEquityWeights(&ctx, &seats, &equities);
std.debug.print("Winner mask: 0b{b:0>3}\n", .{weights.winner_mask});
std.debug.print("Equities: {d:.2} {d:.2} {d:.2}\n", .{
    equities[0], equities[1], equities[2],
});
// Output:
// Winner mask: 0b111
// Equities: 0.33 0.33 0.33
```

### Hero vs Field

```zig
// Calculate hero's equity against multiple opponents
const hero = poker.parseHand("AhAs");
const villains = [_]u64{
    poker.parseHand("KdKc"),
    poker.parseHand("QhQs"),
    poker.parseHand("JcJd"),
};

const hero_equity = try poker.heroVsFieldMonteCarlo(
    hero,
    &villains,
    &.{},
    50000,
    rng,
    allocator,
);
std.debug.print("AA vs 3 opponents: {d:.1}%\n", .{hero_equity * 100});
```

### Threaded Equity (High Volume)

```zig
// Use multiple CPU cores for large simulations
const result = try poker.threaded(
    aa,
    kk,
    &.{},
    10_000_000,  // 10M simulations
    42,           // Random seed
    allocator,
);
std.debug.print("Equity (10M sims): {d:.4}\n", .{result.equity()});
// Automatically uses optimal thread count
```

### Deck Sampling Utilities

```zig
// Swap-remove deck sampler (no rebuilding 52-card arrays)
var sampler = poker.DeckSampler.initWithMask(poker.parseHand("AhAs"));

const card1 = sampler.draw(rng);
const flop = sampler.drawMask(rng, 3); // Draw three cards as a bitmask
std.debug.print("Remaining cards: {}\n", .{sampler.remainingCards()});
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
