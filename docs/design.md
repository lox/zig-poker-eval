# Design and Architecture

## Overview

The evaluator achieves ~2.0ns per hand on MacBook Pro (M5) through CHD perfect hash tables and SIMD batch processing.

## Hand Representation

Each hand is a 64-bit integer:

```text
bits 0-12:   clubs (13 bits, one per rank)
bits 13-25:  diamonds (13 bits, one per rank)
bits 26-38:  hearts (13 bits, one per rank)
bits 39-51:  spades (13 bits, one per rank)
bits 52-63:  unused (12 bits)
```

## Evaluation Strategy

Two paths based on flush detection:

1. **Non-flush path (>99.6%)**: CHD perfect hash tables
2. **Flush path (<0.4%)**: Direct lookup table

```zig
pub fn evaluateHand(hand: card.Hand) HandRank {
    if (isFlushHand(hand)) {
        const pattern = getFlushPattern(hand);
        return tables.flushLookup(pattern);
    }

    const rpc = computeRpcFromHand(hand);
    return chdLookupScalar(rpc);
}
```

## Non-Flush Path: CHD Perfect Hash

### Rank Pattern Code (RPC)

Non-flush hands reduce to rank patterns - count of each rank (0-4). Encoded in base-5:

```zig
fn computeRpcFromHand(hand: u64) u32 {
    var rank_counts = [_]u8{0} ** 13;

    // Count ranks across suits
    for (0..4) |suit| {
        const suit_mask = @as(u16, @truncate((hand >> (@as(u6, @intCast(suit)) * 13)) & RANK_MASK));
        for (0..13) |rank| {
            if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    // Base-5 encoding (49,205 patterns in 31 bits)
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return rpc;
}
```

### CHD Hash Function

Compress, Hash & Displace algorithm:

```zig
inline fn mix64(x: u64, magic_constant: u64) u64 {
    const h = x *% magic_constant;
    return h ^ (h >> 29);
}

pub inline fn lookup(key: u32, magic_constant: u64, g_array: []const u8, value_table: []const u16, table_size: u32) u16 {
    const h = mix64(@as(u64, key), magic_constant);
    const bucket = @as(u32, @intCast(h >> 51));        // Top 13 bits → 8,192 buckets
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits → 131,072 slots
    const displacement = g_array[bucket];
    const final_index = (base_index + displacement) & (table_size - 1);
    return value_table[final_index];
}
```

**Table sizes:**
- Displacement array: 8,192 bytes
- Value table: 262,144 bytes
- Total: ~267KB (fits in L2 cache)

## Flush Path: Direct Lookup

Extract 13-bit pattern of ranks in flush suit:

```zig
pub fn getFlushPattern(hand: u64) u16 {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK,
        @as(u16, @truncate(hand >> 13)) & RANK_MASK,
        @as(u16, @truncate(hand >> 26)) & RANK_MASK,
        @as(u16, @truncate(hand >> 39)) & RANK_MASK,
    };

    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            return getTop5Ranks(suit_mask);
        }
    }
    return 0;
}
```

**Flush table:** 1,287 patterns → 2.5KB (stays in L1)

## SIMD Batch Processing

### Structure-of-Arrays Layout

Hands transposed for SIMD efficiency:

```zig
fn computeRpcSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]u32 {
    // Extract suits (structure-of-arrays)
    var clubs: [batchSize]u16 = undefined;
    var diamonds: [batchSize]u16 = undefined;
    var hearts: [batchSize]u16 = undefined;
    var spades: [batchSize]u16 = undefined;

    for (hands, 0..) |hand, i| {
        clubs[i] = @as(u16, @truncate((hand >> 0) & RANK_MASK));
        diamonds[i] = @as(u16, @truncate((hand >> 13) & RANK_MASK));
        hearts[i] = @as(u16, @truncate((hand >> 26) & RANK_MASK));
        spades[i] = @as(u16, @truncate((hand >> 39) & RANK_MASK));
    }
```

### Vectorized Rank Counting

```zig
const clubs_v: @Vector(batchSize, u16) = clubs;
const diamonds_v: @Vector(batchSize, u16) = diamonds;
const hearts_v: @Vector(batchSize, u16) = hearts;
const spades_v: @Vector(batchSize, u16) = spades;

var rpc_vec: @Vector(batchSize, u32) = @splat(0);

inline for (0..13) |rank| {
    const rank_bit: @Vector(batchSize, u16) = @splat(@as(u16, 1) << @intCast(rank));
    const zero_vec: @Vector(batchSize, u16) = @splat(0);

    // Vectorized rank counting
    const clubs_has = @select(u8, (clubs_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
    const diamonds_has = @select(u8, (diamonds_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
    const hearts_has = @select(u8, (hearts_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
    const spades_has = @select(u8, (spades_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);

    const rank_count_vec = clubs_has + diamonds_has + hearts_has + spades_has;

    // Vectorized base-5 encoding
    const five_vec: @Vector(batchSize, u32) = @splat(5);
    rpc_vec = rpc_vec * five_vec + @as(@Vector(batchSize, u32), rank_count_vec);
}
```

**Batch sizes:**
- Optimal: 32 hands (best overhead amortization)
- Supported: 2, 4, 8, 16, 32, 64
- Fallback: Scalar for other sizes

## Performance Characteristics

**MacBook Pro (M5):**
- Single evaluation: ~4.9ns per hand
- Batch evaluation (32 hands): ~2.0ns per hand
- Throughput: 500M+ hands/second

**Memory access:**
1. CHD displacement lookup: 1 byte from 8KB (L1 hit)
2. CHD value lookup: 2 bytes from 256KB (L2 hit)
3. Working set: ~395KB (267KB CHD + 128KB flush patterns)

**Critical path:**
1. Suit extraction: 4 shifts + 4 masks
2. Rank counting: 52 iterations (13 ranks × 4 suits)
3. Base-5 encoding: 13 multiply-adds
4. CHD hash: 1 multiply + 2 shifts + 1 XOR
5. Table lookups: 2 memory accesses

SIMD amortizes steps 1-3 across multiple hands.

## Equity Calculations

The equity module provides both Monte Carlo simulation and exact enumeration for calculating hand equity.

### API Design

**Unified Result Type:**
All equity functions return `EquityResult` with optional category tracking:

```zig
pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,
    hand1_categories: ?HandCategories = null,  // Populated by detailed variants
    hand2_categories: ?HandCategories = null,  // Populated by detailed variants

    pub fn equity(self: EquityResult) f64 { ... }
    pub fn winRate(self: EquityResult) f64 { ... }
    pub fn tieRate(self: EquityResult) f64 { ... }
    pub fn confidenceInterval(self: EquityResult) ?struct { lower: f64, upper: f64 } { ... }
};
```

**Monte Carlo Functions:**
- `monteCarlo()`: Basic simulation without category tracking
- `detailedMonteCarlo()`: With hand category frequencies and confidence intervals

**Exact Enumeration:**
- `exact()`: Deterministic calculation enumerating all board completions
- `exactDetailed()`: With hand category frequencies

**Implementation:**
Internal `*Impl()` functions use `comptime` branching for zero-cost abstraction:

```zig
fn monteCarloImpl(comptime track_categories: bool, ...) !EquityResult {
    var hand1_cat_storage = HandCategories{};
    var hand2_cat_storage = HandCategories{};

    // ... simulation loop ...

    if (track_categories) {
        hand1_cat_storage.addHand(category);  // Eliminated at compile time if false
    }

    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = simulations,
        .hand1_categories = if (track_categories) hand1_cat_storage else null,
        .hand2_categories = if (track_categories) hand2_cat_storage else null,
    };
}

pub fn monteCarlo(...) !EquityResult {
    return monteCarloImpl(false, ...);  // Category tracking compiled out
}

pub fn detailedMonteCarlo(...) !EquityResult {
    return monteCarloImpl(true, ...);   // Category tracking included
}
```

This design provides:
- Single source of truth (no code duplication)
- Type-safe API (separate functions, not runtime bool parameter)
- Zero runtime cost (comptime branches eliminated)
- Backward compatibility (type aliases for old names)

### Multiway Showdown Helpers

- `evaluateShowdownMultiway(board_ctx, seats[])` processes every seat in a single SIMD-driven pass.
  - Returns `MultiwayResult{ best_rank, winner_mask, tie_count }` so callers can branch-free update win/tie counters.
  - Winner mask uses one bit per seat (supports up to 64 players; production max is 6) which maps cleanly to table indices.
  - Internally reuses the existing batch pipeline (`evaluateBatch`) to keep the hot loop identical to the optimized showdown path.
- `evaluateEquityWeights(board_ctx, seats[], equities[])` builds on the same pass to emit normalized shares (`1 / winner_count`) per seat.
  - Callers can sum the returned weights for Monte Carlo equity without doing per-seat comparisons or managing temporary rank buffers.
  - The function returns the underlying `MultiwayResult`, so consumers can still access the best rank or bitmask when they need discrete outcomes.

### Deck Sampling Utilities

- `FULL_DECK` exposes the canonical 52-card bitfield array for anyone who needs direct indexed access.
- `DeckSampler` provides O(1) swap-remove draws driven by an internal index map:
  - `resetWithMask(exclude)` and `initWithMask(exclude)` exclude known cards in O(k).
  - `draw()` and `drawMask(count)` deliver single cards or aggregated bitmasks without rebuilding temporary 52-entry arrays.
  - `removeMask`/`removeCard` update the index map in constant time, enabling cheap reuse across samples.
  - Designed to pair with any RNG that implements `uintLessThan` (e.g. `std.rand`), so external callers can plug in their preferred generator.

## Component Diagram

```
┌─────────────────────────────────────┐
│         poker.zig (Public API)      │
└─────────────────┬───────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼──────┐      ┌────────▼─────────┐
│ evaluator  │      │  Game Components │
│ ─────────  │      │  ───────────────  │
│ Scalar     │      │ card.zig         │
│ SIMD batch │      │ hand.zig         │
│ RPC        │      │ range.zig        │
│ Flush      │      │ equity.zig       │
└─────┬──────┘      │ analysis.zig     │
      │             │ draws.zig        │
      │             └──────────────────┘
      │
┌─────▼────────────────────┐
│ internal/                │
│ ────────                 │
│ tables.zig (lookup)     │
│ mphf.zig (CHD)          │
│ slow_evaluator.zig      │
│ build_tables.zig        │
└─────────────────────────┘
```

## Key Insights

**What worked:**
1. Simple nested loops - compiler optimizes well
2. SIMD batching - true algorithmic parallelism
3. Perfect hash tables - predictable memory access
4. Two-path design - minimal overhead for common case

**What didn't work:**
1. ARM64 packed tables - bit manipulation overhead
2. Prefetching - tables already cache-resident
3. Complex RPC computation - simple loops faster
4. 8-hand batching - register pressure

**Memory-bound → Compute-bound:** With 395KB tables in L2, bottleneck is computation, not memory.

## Correctness

- All 133M possible 7-card hands verified
- Fast evaluator matches slow reference implementation
- Edge cases: two trips → full house, wheel straights
- Zig's compile-time guarantees prevent many bugs

## Build System Integration

Lookup tables pre-generated and compiled into binary:

```bash
# One-time table generation (~20 seconds)
zig build build-tables -Doptimize=ReleaseFast

# Creates src/internal/tables.zig (~395KB of data)
```

Zero startup overhead, optimal memory layout.
