# Zig Poker Evaluator - Implementation Design

This document describes the actual implementation of the Zig poker evaluator, which achieves ~4.5ns per hand evaluation on Apple M1 through a combination of perfect hash tables and SIMD batch processing.

## 1. Core Architecture

### 1.1 Hand Representation

Each poker hand is represented as a 64-bit integer using a bitfield layout:

```text
bits 0-12:   clubs (13 bits, one per rank)
bits 13-25:  diamonds (13 bits, one per rank)
bits 26-38:  hearts (13 bits, one per rank)
bits 39-51:  spades (13 bits, one per rank)
bits 52-63:  unused (12 bits)
```

This representation allows efficient suit extraction and bit manipulation operations.

### 1.2 Two-Path Evaluation Strategy

The evaluator uses a two-path approach based on flush detection:

1. **Non-flush path (>99.6% of hands)**: Uses CHD perfect hash tables
2. **Flush path (<0.4% of hands)**: Uses a smaller direct lookup table

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

## 2. Non-Flush Path: CHD Perfect Hash

### 2.1 Rank Pattern Code (RPC) Encoding

Non-flush hands are reduced to their rank pattern - a count of how many times each rank appears (0-4 times). This is encoded using base-5 arithmetic to fit in 31 bits:

```zig
fn computeRpcFromHand(hand: u64) u32 {
    var rank_counts = [_]u8{0} ** 13;

    // Count occurrences of each rank across all suits
    for (0..4) |suit| {
        const suit_mask = @as(u16, @truncate((hand >> (@as(u6, @intCast(suit)) * 13)) & RANK_MASK));
        for (0..13) |rank| {
            if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    // Base-5 encoding: preserves all 49,205 patterns in 31 bits
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return rpc;
}
```

### 2.2 CHD Hash Function

The CHD (Compress, Hash & Displace) algorithm uses a two-level hash:

```zig
inline fn mix64(x: u64, magic_constant: u64) u64 {
    const h = x *% magic_constant;
    return h ^ (h >> 29);
}

pub inline fn lookup(key: u32, magic_constant: u64, g_array: []const u8, value_table: []const u16, table_size: u32) u16 {
    const h = mix64(@as(u64, key), magic_constant);
    const bucket = @as(u32, @intCast(h >> 51));        // Top 13 bits -> 8,192 buckets
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits -> 131,072 slots
    const displacement = g_array[bucket];
    const final_index = (base_index + displacement) & (table_size - 1);
    return value_table[final_index];
}
```

### 2.3 Table Sizes

- **Displacement array (g)**: 8,192 bytes (one u8 per bucket)
- **Value table**: 262,144 bytes (131,072 × 2 bytes)
- **Total**: ~267 KB (fits in L2 cache)

## 3. Flush Path: Direct Lookup

### 3.1 Flush Pattern Extraction

For flush hands, we extract the 13-bit pattern of ranks in the flush suit:

```zig
pub fn getFlushPattern(hand: u64) u16 {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK,   // clubs
        @as(u16, @truncate(hand >> 13)) & RANK_MASK,  // diamonds
        @as(u16, @truncate(hand >> 26)) & RANK_MASK,  // hearts
        @as(u16, @truncate(hand >> 39)) & RANK_MASK,  // spades
    };

    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            return getTop5Ranks(suit_mask);
        }
    }
    return 0;
}
```

### 3.2 Flush Table

The flush lookup table directly maps the 1,287 possible 5-card flush patterns to their hand ranks. This small table (2.5 KB) stays hot in L1 cache.

## 4. SIMD Batch Processing

### 4.1 Structure-of-Arrays Layout

For SIMD efficiency, hands are transposed from array-of-structures to structure-of-arrays:

```zig
fn computeRpcSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]u32 {
    // Extract suits for all hands (structure-of-arrays)
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

### 4.2 Vectorized Rank Counting

The rank counting loop is fully vectorized using Zig's `@Vector` types:

```zig
const clubs_v: @Vector(batchSize, u16) = clubs;
const diamonds_v: @Vector(batchSize, u16) = diamonds;
const hearts_v: @Vector(batchSize, u16) = hearts;
const spades_v: @Vector(batchSize, u16) = spades;

var rpc_vec: @Vector(batchSize, u32) = @splat(0);

inline for (0..13) |rank| {
    const rank_bit: @Vector(batchSize, u16) = @splat(@as(u16, 1) << @intCast(rank));
    const zero_vec: @Vector(batchSize, u16) = @splat(0);

    // Count rank occurrences across all suits (vectorized)
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

### 4.3 Batch Sizes

The implementation supports various batch sizes with SIMD optimization:

- **Optimal**: 32 hands (default) - best amortization of overhead
- **Supported**: 2, 4, 8, 16, 32, 64 hands
- **Fallback**: Scalar processing for other sizes

## 5. Performance Characteristics

### 5.1 Measured Performance (Apple M1)

- **Single evaluation**: ~4.5ns per hand
- **Batch evaluation (32 hands)**: ~4.1ns per hand
- **Throughput**: 224M+ hands/second

### 5.2 Memory Access Pattern

1. **CHD displacement lookup**: 1 byte from 8KB array (L1 hit)
2. **CHD value lookup**: 2 bytes from 256KB array (L2 hit)
3. **Total working set**: ~267KB (comfortably fits in 512KB L2)

### 5.3 Critical Path Analysis

The scalar evaluation critical path:

1. Suit extraction: 4 shifts + 4 masks (parallelizable)
2. Rank counting: 52 iterations (13 ranks × 4 suits)
3. Base-5 encoding: 13 multiply-adds
4. CHD hash: 1 multiply + 2 shifts + 1 XOR
5. Table lookups: 2 memory accesses

SIMD amortizes steps 1-3 across multiple hands, achieving near-linear speedup.

## 6. Implementation Trade-offs

### 6.1 What Worked

1. **Simple nested loops**: Compiler optimizes better than manual unrolling
2. **SIMD batching**: True algorithmic parallelism beats micro-optimizations
3. **Perfect hash tables**: Predictable memory access patterns
4. **Two-path design**: Minimal overhead for common case (non-flush)

### 6.2 What Didn't Work

From experiments (see experiments.md):

1. **ARM64 packed tables**: Bit manipulation overhead exceeded memory savings
2. **Prefetching**: Tables already cache-resident
3. **Complex RPC computation**: LUT approaches slower than simple loops
4. **8-hand batching**: Register pressure outweighed benefits

### 6.3 Key Insights

1. **Memory-bound → Compute-bound**: With 267KB tables in L2, the bottleneck is computation, not memory
2. **Compiler optimization**: Zig/LLVM's auto-vectorization often beats manual optimization
3. **Architecture matters**: 4-wide SIMD is optimal for ARM64 NEON
4. **Measurement artifacts**: Heavy instrumentation can show 3x overhead

## 7. Correctness Guarantees

The implementation maintains 100% accuracy through:

1. **Comprehensive testing**: All 133M possible 7-card hands verified
2. **Cross-validation**: Fast evaluator matches slow reference implementation
3. **Edge case handling**: Two trips → full house, wheel straights, etc.
4. **Type safety**: Zig's compile-time guarantees prevent many bugs

## 8. Build System Integration

The lookup tables are pre-generated and compiled into the binary:

```bash
# One-time table generation (requires ~20 seconds)
zig build build-tables -Doptimize=ReleaseFast

# This creates src/internal/tables.zig with ~270KB of data
```

The main build then includes these tables as compile-time constants, ensuring zero startup overhead and optimal memory layout.
