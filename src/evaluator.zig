const std = @import("std");
const card = @import("card");

// Internal dependencies
const tables = @import("internal/tables.zig");
const slow_eval = @import("internal/slow_evaluator.zig");

/// Hand rank type - lower numbers represent stronger hands
pub const HandRank = u16;

/// Hand categories from weakest to strongest
pub const HandCategory = enum(u4) {
    high_card = 1,
    pair = 2,
    two_pair = 3,
    three_of_a_kind = 4,
    straight = 5,
    flush = 6,
    full_house = 7,
    four_of_a_kind = 8,
    straight_flush = 9,
};

/// Convert evaluator rank (lower=better) to HandCategory enum
pub fn getHandCategory(rank: HandRank) HandCategory {
    const category_index = rank / slow_eval.CATEGORY_STEP;
    return switch (category_index) {
        0 => .straight_flush,
        1 => .four_of_a_kind,
        2 => .full_house,
        3 => .flush,
        4 => .straight,
        5 => .three_of_a_kind,
        6 => .two_pair,
        7 => .pair,
        else => .high_card,
    };
}

// === Core Constants ===

const RANK_MASK = 0x1FFF; // 13 bits for ranks

// === Flush Pattern Lookup Table ===

/// Compile-time lookup table for extracting top 5 ranks from flush suits
/// Maps any u16 suit mask to its top 5 ranks (for 5-7 card flushes)
/// Table size: 65,536 entries * 2 bytes = 128KB
const flush_top5_table: [65536]u16 = blk: {
    @setEvalBranchQuota(2000000); // Need higher quota for 65K table generation
    var table: [65536]u16 = [_]u16{0} ** 65536;

    // Compute top 5 pattern for every possible suit mask
    for (0..65536) |mask_int| {
        const suit_mask: u16 = @intCast(mask_int);
        const bit_count = @popCount(suit_mask);

        // Only valid for 5-7 card flushes
        if (bit_count < 5 or bit_count > 7) {
            table[mask_int] = 0; // Invalid input, never called
            continue;
        }

        // Fast path: exactly 5 bits set
        if (bit_count == 5) {
            table[mask_int] = suit_mask;
            continue;
        }

        // Check for straights
        const straights = [_]u16{
            0x1F00, 0x0F80, 0x07C0, 0x03E0, 0x01F0,
            0x00F8, 0x007C, 0x003E, 0x001F, 0x100F, // wheel (A-2-3-4-5)
        };

        var is_straight = false;
        for (straights) |pattern| {
            if ((suit_mask & pattern) == pattern) {
                table[mask_int] = pattern;
                is_straight = true;
                break;
            }
        }

        if (is_straight) continue;

        // Take highest 5 ranks
        var result: u16 = 0;
        var count: u8 = 0;
        var rank: i8 = 12;

        while (count < 5 and rank >= 0) : (rank -= 1) {
            const bit = @as(u16, 1) << @intCast(rank);
            if ((suit_mask & bit) != 0) {
                result |= bit;
                count += 1;
            }
        }

        table[mask_int] = result;
    }

    break :blk table;
};

/// Cached board analysis for reusing evaluation work across multiple hole cards
pub const BoardContext = struct {
    board: card.Hand,
    suit_masks: [4]u16,
    suit_counts: [4]u8,
    rank_counts: [13]u8,
    rpc_base: u32, // Cached board-only RPC for incremental calculation (Exp 21)
    suit_flush_mask_ge3: u8, // Bitmask of suits with ≥3 cards (flush candidates)
};

fn initSuitMasks(board: card.Hand) [4]u16 {
    var suits: [4]u16 = undefined;
    inline for (0..4) |suit| {
        const offset: u6 = @intCast(suit * 13);
        suits[suit] = @as(u16, @truncate((board >> offset) & RANK_MASK));
    }
    return suits;
}

fn computeBoardRankCounts(suit_masks: [4]u16) [13]u8 {
    var counts = [_]u8{0} ** 13;
    inline for (0..13) |rank| {
        var total: u8 = 0;
        inline for (0..4) |suit| {
            if ((suit_masks[suit] & (@as(u16, 1) << @intCast(rank))) != 0) {
                total += 1;
            }
        }
        counts[rank] = total;
    }
    return counts;
}

/// Pre-compute board state for fast showdown reuse
pub fn initBoardContext(board: card.Hand) BoardContext {
    const suit_masks = initSuitMasks(board);
    var suit_counts: [4]u8 = undefined;
    inline for (0..4) |suit| {
        suit_counts[suit] = @intCast(@popCount(suit_masks[suit]));
    }

    const rank_counts = computeBoardRankCounts(suit_masks);

    // Exp 21: Precompute board-only RPC to enable O(1) incremental updates
    var rpc_base: u32 = 0;
    for (rank_counts) |count| {
        rpc_base = rpc_base * 5 + count;
    }

    // Exp 21: Mark suits that can reach 5 cards (≥3 on board + 2 hole cards)
    var suit_flush_mask_ge3: u8 = 0;
    inline for (0..4) |suit| {
        if (suit_counts[suit] >= 3) {
            suit_flush_mask_ge3 |= @as(u8, 1) << @intCast(suit);
        }
    }

    return .{
        .board = board,
        .suit_masks = suit_masks,
        .suit_counts = suit_counts,
        .rank_counts = rank_counts,
        .rpc_base = rpc_base,
        .suit_flush_mask_ge3 = suit_flush_mask_ge3,
    };
}

// Exp 21: Compile-time lookup table for base-5 powers used in incremental RPC
// Powers are [5^12, 5^11, ..., 5^1, 5^0] indexed by rank [0..12]
const rpc_powers = blk: {
    @setEvalBranchQuota(2000);
    var powers: [13]u32 = undefined;
    for (0..13) |rank| {
        var pow: u32 = 1;
        var i: usize = 0;
        while (i < (12 - rank)) : (i += 1) {
            pow *= 5;
        }
        powers[rank] = pow;
    }
    break :blk powers;
};

fn evaluateHoleWithContextImpl(ctx: *const BoardContext, hole: card.Hand) HandRank {
    std.debug.assert((hole & ctx.board) == 0);

    var suit_masks = ctx.suit_masks;
    var suit_counts = ctx.suit_counts;
    var rpc_delta: u32 = 0;

    // Exp 21: Process hole cards and accumulate RPC delta incrementally
    var remaining = hole;
    while (remaining != 0) {
        const bit_index: u6 = @intCast(@ctz(remaining));
        remaining &= remaining - 1;
        const suit_index: usize = @intCast(bit_index / 13);
        const rank_index: usize = @intCast(bit_index % 13);

        suit_masks[suit_index] |= @as(u16, 1) << @intCast(rank_index);
        suit_counts[suit_index] += 1;
        rpc_delta += rpc_powers[rank_index]; // O(1) lookup vs O(13) loop
    }

    // Exp 21: Gated flush check - only test suits that can reach 5 cards
    const flush_candidate_mask = ctx.suit_flush_mask_ge3;
    inline for (0..4) |suit| {
        if ((flush_candidate_mask & (@as(u8, 1) << @intCast(suit))) != 0) {
            if (suit_counts[suit] >= 5) {
                const pattern = getTop5Ranks(suit_masks[suit]);
                return tables.flushLookup(pattern);
            }
        }
    }

    // Exp 21: Incremental RPC - add hole card delta to cached board base
    const rpc = ctx.rpc_base + rpc_delta;
    return tables.lookup(rpc);
}

/// Evaluate a pair of hole cards using a precomputed board context
pub fn evaluateHoleWithContext(ctx: *const BoardContext, hole: card.Hand) HandRank {
    return evaluateHoleWithContextImpl(ctx, hole);
}

/// Evaluate a showdown using shared board context
///
/// Experiment 20: SIMD micro-batch optimization
/// Uses batch-2 SIMD path instead of two serial evaluations to:
/// - Enable memory-level parallelism for CHD lookups
/// - Leverage SIMD vectorization for RPC and flush detection
/// - Eliminate redundant array copies in scalar path
/// Expected: 2-2.5× faster than scalar path
pub fn evaluateShowdownWithContext(ctx: *const BoardContext, hero_hole: card.Hand, villain_hole: card.Hand) i8 {
    std.debug.assert((hero_hole & villain_hole) == 0);
    std.debug.assert((hero_hole & ctx.board) == 0);
    std.debug.assert((villain_hole & ctx.board) == 0);

    // Pack both hands into batch-2 vector for SIMD evaluation
    const hands = @Vector(2, u64){
        ctx.board | hero_hole,
        ctx.board | villain_hole,
    };

    const ranks = evaluateBatch(2, hands);
    return if (ranks[0] < ranks[1]) @as(i8, 1) else if (ranks[0] > ranks[1]) @as(i8, -1) else @as(i8, 0);
}

fn showdownChunk(comptime batch_size: usize, ctx: *const BoardContext, hero_holes: []const card.Hand, villain_holes: []const card.Hand, results: []i8) void {
    var hero_array: [batch_size]u64 = undefined;
    var villain_array: [batch_size]u64 = undefined;

    inline for (0..batch_size) |i| {
        const hero_hole = hero_holes[i];
        const villain_hole = villain_holes[i];
        std.debug.assert((hero_hole & villain_hole) == 0);
        std.debug.assert((hero_hole & ctx.board) == 0);
        std.debug.assert((villain_hole & ctx.board) == 0);

        hero_array[i] = ctx.board | hero_hole;
        villain_array[i] = ctx.board | villain_hole;
    }

    const hero_vec: @Vector(batch_size, u64) = hero_array;
    const villain_vec: @Vector(batch_size, u64) = villain_array;

    const hero_ranks = evaluateBatch(batch_size, hero_vec);
    const villain_ranks = evaluateBatch(batch_size, villain_vec);

    inline for (0..batch_size) |i| {
        results[i] = if (hero_ranks[i] < villain_ranks[i])
            @as(i8, 1)
        else if (hero_ranks[i] > villain_ranks[i])
            @as(i8, -1)
        else
            @as(i8, 0);
    }
}

/// Evaluate many hero/villain pairs that share the same board context.
///
/// `hero_holes` and `villain_holes` must have the same length and represent
/// disjoint two-card combinations relative to the board. Results are stored as
/// -1 (villain wins), 0 (tie), 1 (hero wins).
pub fn evaluateShowdownBatch(
    ctx: *const BoardContext,
    hero_holes: []const card.Hand,
    villain_holes: []const card.Hand,
    results: []i8,
) void {
    std.debug.assert(hero_holes.len == villain_holes.len);
    std.debug.assert(results.len >= hero_holes.len);

    var index: usize = 0;
    const total = hero_holes.len;

    while (index + 32 <= total) {
        showdownChunk(32, ctx, hero_holes[index .. index + 32], villain_holes[index .. index + 32], results[index .. index + 32]);
        index += 32;
    }
    if (index + 16 <= total) {
        showdownChunk(16, ctx, hero_holes[index .. index + 16], villain_holes[index .. index + 16], results[index .. index + 16]);
        index += 16;
    }
    if (index + 8 <= total) {
        showdownChunk(8, ctx, hero_holes[index .. index + 8], villain_holes[index .. index + 8], results[index .. index + 8]);
        index += 8;
    }
    if (index + 4 <= total) {
        showdownChunk(4, ctx, hero_holes[index .. index + 4], villain_holes[index .. index + 4], results[index .. index + 4]);
        index += 4;
    }
    if (index + 2 <= total) {
        showdownChunk(2, ctx, hero_holes[index .. index + 2], villain_holes[index .. index + 2], results[index .. index + 2]);
        index += 2;
    }
    if (index < total) {
        showdownChunk(1, ctx, hero_holes[index .. index + 1], villain_holes[index .. index + 1], results[index .. index + 1]);
    }
}

/// Aggregate result for evaluating multiple seats in a single pass.
pub const MultiwayResult = struct {
    best_rank: HandRank,
    winner_mask: u64,
    tie_count: u8,
};

fn multiwayChunk(comptime batch_size: usize, ctx: *const BoardContext, holes: []const card.Hand, base_index: usize, acc: *MultiwayResult) void {
    std.debug.assert(holes.len == batch_size);

    var hands_array: [batch_size]u64 = undefined;
    inline for (0..batch_size) |i| {
        const hole = holes[i];
        std.debug.assert((hole & ctx.board) == 0);
        hands_array[i] = ctx.board | hole;
    }

    const hands_vec: @Vector(batch_size, u64) = hands_array;
    const ranks = evaluateBatch(batch_size, hands_vec);

    var best_rank = acc.*.best_rank;
    var winner_mask = acc.*.winner_mask;
    var tie_count = acc.*.tie_count;

    inline for (0..batch_size) |i| {
        const global_index = base_index + i;
        std.debug.assert(global_index < 64);
        const rank = ranks[i];

        if (rank < best_rank) {
            best_rank = rank;
            winner_mask = @as(u64, 1) << @intCast(global_index);
            tie_count = 1;
        } else if (rank == best_rank) {
            winner_mask |= @as(u64, 1) << @intCast(global_index);
            tie_count += 1;
        }
    }

    acc.* = .{
        .best_rank = best_rank,
        .winner_mask = winner_mask,
        .tie_count = tie_count,
    };
}

/// Evaluate all seats in `holes` simultaneously, returning the best rank,
/// a bitmask of winning seats, and the number of winners.
pub fn evaluateShowdownMultiway(ctx: *const BoardContext, holes: []const card.Hand) MultiwayResult {
    std.debug.assert(holes.len > 0);
    std.debug.assert(holes.len <= 64);

    var used_cards: card.Hand = ctx.board;
    for (holes) |hole| {
        std.debug.assert(card.countCards(hole) == 2);
        std.debug.assert((hole & ctx.board) == 0);
        std.debug.assert((hole & used_cards) == 0);
        used_cards |= hole;
    }

    var acc = MultiwayResult{
        .best_rank = std.math.maxInt(HandRank),
        .winner_mask = 0,
        .tie_count = 0,
    };

    var index: usize = 0;
    const total = holes.len;

    while (index + 32 <= total) {
        multiwayChunk(32, ctx, holes[index .. index + 32], index, &acc);
        index += 32;
    }
    if (index + 16 <= total) {
        multiwayChunk(16, ctx, holes[index .. index + 16], index, &acc);
        index += 16;
    }
    if (index + 8 <= total) {
        multiwayChunk(8, ctx, holes[index .. index + 8], index, &acc);
        index += 8;
    }
    if (index + 4 <= total) {
        multiwayChunk(4, ctx, holes[index .. index + 4], index, &acc);
        index += 4;
    }
    if (index + 2 <= total) {
        multiwayChunk(2, ctx, holes[index .. index + 2], index, &acc);
        index += 2;
    }
    if (index < total) {
        multiwayChunk(1, ctx, holes[index .. index + 1], index, &acc);
    }

    std.debug.assert(acc.tie_count > 0);
    return acc;
}

/// Evaluate seats and produce per-player normalized equities (1/tie_count for winners).
/// Returns the same `MultiwayResult` that powers the weight calculation.
pub fn evaluateEquityWeights(ctx: *const BoardContext, holes: []const card.Hand, equities: []f64) MultiwayResult {
    std.debug.assert(equities.len >= holes.len);

    const result = evaluateShowdownMultiway(ctx, holes);

    const slice = equities[0..holes.len];
    @memset(slice, 0.0);

    if (result.tie_count != 0) {
        const share = 1.0 / @as(f64, @floatFromInt(result.tie_count));
        var mask = result.winner_mask;
        while (mask != 0) {
            const bit_index: usize = @intCast(@ctz(mask));
            std.debug.assert(bit_index < holes.len);
            slice[bit_index] = share;
            mask &= mask - 1;
        }
    }

    return result;
}

// === Scalar RPC Computation ===

fn computeRpcFromHand(hand: u64) u32 {
    var rank_counts = [_]u8{0} ** 13;

    for (0..4) |suit| {
        const suit_mask = @as(u16, @truncate((hand >> (@as(u6, @intCast(suit)) * 13)) & RANK_MASK));
        for (0..13) |rank| {
            if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    // Base-5 encoding: preserves all patterns in 31 bits
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return rpc;
}

// === SIMD RPC Computation ===

fn computeRpcSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]u32 {
    var result: [batchSize]u32 = undefined;

    // Use SIMD for batch sizes that are powers of 2 or have good SIMD support
    // Note: Requires LLVM backend on x86_64 due to self-hosted backend bug with vpmovzxbd instruction
    if (batchSize == 2 or batchSize == 4 or batchSize == 8 or batchSize == 16 or batchSize == 32 or batchSize == 64) {
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

        const clubs_v: @Vector(batchSize, u16) = clubs;
        const diamonds_v: @Vector(batchSize, u16) = diamonds;
        const hearts_v: @Vector(batchSize, u16) = hearts;
        const spades_v: @Vector(batchSize, u16) = spades;

        var rpc_vec: @Vector(batchSize, u32) = @splat(0);

        // Vectorized rank counting for all 13 ranks
        inline for (0..13) |rank| {
            const rank_bit: @Vector(batchSize, u16) = @splat(@as(u16, 1) << @intCast(rank));
            const zero_vec: @Vector(batchSize, u16) = @splat(0);

            // Count rank occurrences across all suits (vectorized)
            const one_vec: @Vector(batchSize, u8) = @splat(1);
            const zero_u8_vec: @Vector(batchSize, u8) = @splat(0);

            const clubs_has = @select(u8, (clubs_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
            const diamonds_has = @select(u8, (diamonds_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
            const hearts_has = @select(u8, (hearts_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
            const spades_has = @select(u8, (spades_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);

            // Sum to get rank count for each hand
            const rank_count_vec = clubs_has + diamonds_has + hearts_has + spades_has;

            // Vectorized base-5 encoding: rpc = rpc * 5 + count
            const five_vec: @Vector(batchSize, u32) = @splat(5);
            rpc_vec = rpc_vec * five_vec + @as(@Vector(batchSize, u32), rank_count_vec);
        }

        const rpc_array: [batchSize]u32 = @as([batchSize]u32, rpc_vec);
        @memcpy(&result, &rpc_array);
    } else {
        // Fallback to scalar computation for other batch sizes
        for (hands, 0..) |hand, i| {
            result[i] = computeRpcFromHand(hand);
        }
    }

    return result;
}

inline fn chdLookupScalar(rpc: u32) u16 {
    return tables.lookup(rpc);
}

// === SIMD Flush Detection ===

// NOTE: Experiment 14 attempted to return suit data alongside flush mask to eliminate
// redundant suit extraction in getFlushPattern(). This failed (1.8% slower) because:
// - Stack allocation overhead (256 bytes for batch-32) exceeded computational savings
// - Storing suits for all hands when only ~20% are flushes is wasteful
// - "Extract on demand" is more efficient than "extract once, cache everywhere"
// See docs/experiments.md Experiment 14 for detailed analysis.

/// Detect flush hands in a batch using SIMD
fn detectFlushSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]bool {
    var result: [batchSize]bool = [_]bool{false} ** batchSize;

    if (batchSize == 2 or batchSize == 4 or batchSize == 8 or batchSize == 16 or batchSize == 32 or batchSize == 64) {
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

        // Vectorized popcount for each suit
        const clubs_v: @Vector(batchSize, u16) = clubs;
        const diamonds_v: @Vector(batchSize, u16) = diamonds;
        const hearts_v: @Vector(batchSize, u16) = hearts;
        const spades_v: @Vector(batchSize, u16) = spades;

        const clubs_count = @popCount(clubs_v);
        const diamonds_count = @popCount(diamonds_v);
        const hearts_count = @popCount(hearts_v);
        const spades_count = @popCount(spades_v);

        const threshold: @Vector(batchSize, u16) = @splat(5);

        // Check if any suit has >= 5 cards
        const clubs_flush = clubs_count >= threshold;
        const diamonds_flush = diamonds_count >= threshold;
        const hearts_flush = hearts_count >= threshold;
        const spades_flush = spades_count >= threshold;

        // Combine results (any suit with >= 5 means flush) - use bitwise OR for vectors
        const has_flush = clubs_flush | diamonds_flush | hearts_flush | spades_flush;

        // Convert vector to array
        inline for (0..batchSize) |i| {
            result[i] = has_flush[i];
        }
    } else {
        // Fallback to scalar detection
        for (hands, 0..) |hand, i| {
            result[i] = isFlushHand(hand);
        }
    }

    return result;
}

// === Flush Detection and Pattern Extraction ===

pub inline fn isFlushHand(hand: u64) bool {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK, // clubs
        @as(u16, @truncate(hand >> 13)) & RANK_MASK, // diamonds
        @as(u16, @truncate(hand >> 26)) & RANK_MASK, // hearts
        @as(u16, @truncate(hand >> 39)) & RANK_MASK, // spades
    };

    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) return true;
    }
    return false;
}

pub fn getFlushPattern(hand: u64) u16 {
    // NOTE: Yes, this re-extracts suits that were already extracted in detectFlushSimd.
    // Experiment 14 showed that caching suit data to avoid this redundancy actually
    // hurts performance (-1.8%) due to struct allocation overhead exceeding the savings.
    // The compiler optimizes this "redundant" extraction better than manual caching.
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK, // clubs
        @as(u16, @truncate(hand >> 13)) & RANK_MASK, // diamonds
        @as(u16, @truncate(hand >> 26)) & RANK_MASK, // hearts
        @as(u16, @truncate(hand >> 39)) & RANK_MASK, // spades
    };

    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            return getTop5Ranks(suit_mask);
        }
    }
    return 0; // Should never happen for flush hands
}

/// Extract top 5 ranks from a flush suit using lookup table
/// Eliminates branching and iteration - single memory load
inline fn getTop5Ranks(suit_mask: u16) u16 {
    return flush_top5_table[suit_mask];
}

// === Public API ===

/// Evaluate a single 7-card hand and return its rank
/// Lower ranks represent stronger hands (0 = royal flush, 7461 = worst high card)
pub fn evaluateHand(hand: card.Hand) HandRank {
    if (isFlushHand(hand)) {
        const pattern = getFlushPattern(hand);
        return tables.flushLookup(pattern);
    }

    const rpc = computeRpcFromHand(hand);
    return chdLookupScalar(rpc);
}

/// Evaluate a batch of hands with configurable batch size
/// batchSize must be known at compile time for optimal performance
pub fn evaluateBatch(comptime batchSize: usize, hands: @Vector(batchSize, u64)) @Vector(batchSize, u16) {
    // Convert vector to array for processing
    var hands_array: [batchSize]u64 = undefined;
    inline for (0..batchSize) |i| {
        hands_array[i] = hands[i];
    }

    // Compute RPC for all hands (SIMD-optimized)
    const rpc_results = computeRpcSimd(batchSize, &hands_array);

    // Detect flush hands in batch (SIMD-optimized)
    const flush_mask = detectFlushSimd(batchSize, &hands_array);

    // Evaluate each hand using pre-computed flush detection
    var results: [batchSize]u16 = undefined;
    inline for (0..batchSize) |i| {
        if (flush_mask[i]) {
            results[i] = tables.flushLookup(getFlushPattern(hands_array[i]));
        } else {
            results[i] = chdLookupScalar(rpc_results[i]);
        }
    }

    return @as(@Vector(batchSize, u16), results);
}

/// Default batch size for optimal performance on modern CPUs
pub const DEFAULT_BATCH_SIZE = 32;

/// Evaluate hands using the optimal batch size
pub fn evaluateBatch32(hands: @Vector(32, u64)) @Vector(32, u16) {
    return evaluateBatch(32, hands);
}

// === Benchmarking Functions ===

pub fn benchmarkSingle(iterations: u32) u64 {
    var sum: u64 = 0;
    const test_hand: u64 = 0x123456789ABCD;

    for (0..iterations) |_| {
        sum +%= evaluateHand(test_hand);
    }

    return sum;
}

pub fn benchmarkBatch(iterations: u32) u64 {
    var sum: u64 = 0;
    const test_hands = @Vector(4, u64){ 0x1F00000000000, 0x123456789ABCD, 0x0F0F0F0F0F0F0, 0x1F00 };

    for (0..iterations) |_| {
        const results = evaluateBatch(4, test_hands);
        sum +%= results[0] + results[1] + results[2] + results[3];
    }

    return sum;
}

/// Benchmark different batch sizes
pub fn benchmarkBatchSize(comptime batchSize: usize, iterations: u32) u64 {
    var sum: u64 = 0;

    // Generate test hands
    var test_hands_array: [batchSize]u64 = undefined;
    for (0..batchSize) |i| {
        test_hands_array[i] = 0x123456789ABCD +% (i * 0x1000);
    }
    const test_hands: @Vector(batchSize, u64) = test_hands_array;

    for (0..iterations) |_| {
        const results = evaluateBatch(batchSize, test_hands);

        // Sum all results
        inline for (0..batchSize) |i| {
            sum +%= results[i];
        }
    }

    return sum;
}

// === Test Utilities ===

/// Expose slow evaluator for validation
pub const slow_evaluator = slow_eval;

/// Generate a batch of random 7-card hands with comptime size
pub fn generateRandomHandBatch(comptime size: usize, rng: *std.Random) @Vector(size, u64) {
    var hands: [size]u64 = undefined;

    for (&hands) |*hand| {
        hand.* = generateRandomHand(rng);
    }

    return @as(@Vector(size, u64), hands);
}

/// Generate a single random 7-card hand
pub fn generateRandomHand(rng: *std.Random) u64 {
    var hand: u64 = 0;
    var cards_dealt: u8 = 0;

    while (cards_dealt < 7) {
        const suit: card.Suit = @enumFromInt(rng.intRangeAtMost(u8, 0, 3));
        const rank: card.Rank = @enumFromInt(rng.intRangeAtMost(u8, 0, 12));
        const card_bits = card.makeCard(suit, rank);

        if ((hand & card_bits) == 0) {
            hand |= card_bits;
            cards_dealt += 1;
        }
    }

    return hand;
}

// === Tests ===

const testing = std.testing;

test "hand category conversion" {
    const straight_flush = card.makeCard(.clubs, .nine) |
        card.makeCard(.clubs, .eight) |
        card.makeCard(.clubs, .seven) |
        card.makeCard(.clubs, .six) |
        card.makeCard(.clubs, .five) |
        card.makeCard(.diamonds, .ace) |
        card.makeCard(.hearts, .king);
    const straight_flush_rank = evaluateHand(straight_flush);
    try testing.expect(getHandCategory(straight_flush_rank) == .straight_flush);

    const four_kind = card.makeCard(.clubs, .ace) |
        card.makeCard(.diamonds, .ace) |
        card.makeCard(.hearts, .ace) |
        card.makeCard(.spades, .ace) |
        card.makeCard(.clubs, .king) |
        card.makeCard(.diamonds, .queen) |
        card.makeCard(.hearts, .jack);
    const four_kind_rank = evaluateHand(four_kind);
    try testing.expect(getHandCategory(four_kind_rank) == .four_of_a_kind);

    const high_card = card.makeCard(.clubs, .ace) |
        card.makeCard(.diamonds, .queen) |
        card.makeCard(.hearts, .ten) |
        card.makeCard(.spades, .eight) |
        card.makeCard(.clubs, .five) |
        card.makeCard(.diamonds, .three) |
        card.makeCard(.hearts, .two);
    const high_card_rank = evaluateHand(high_card);
    try testing.expect(getHandCategory(high_card_rank) == .high_card);
}

test "flush pattern extraction" {
    // Test royal flush in spades: As Ks Qs Js Ts + 2 non-spade cards
    const royal_flush: u64 =
        (@as(u64, 0x1F00) << 39) | // spades: A K Q J T (bits 12,11,10,9,8)
        (@as(u64, 0x0040) << 26) | // hearts: 7 (bit 6)
        (@as(u64, 0x0020) << 13); // diamonds: 6 (bit 5)

    try testing.expect(isFlushHand(royal_flush));
    const pattern = getFlushPattern(royal_flush);
    try testing.expectEqual(@as(u16, 0x1F00), pattern); // A K Q J T pattern
}

test "straight flush pattern" {
    // Test straight flush 9-5 in clubs: 9c 8c 7c 6c 5c + 2 non-club cards
    const straight_flush: u64 =
        (@as(u64, 0x03E0) << 0) | // clubs: 9 8 7 6 5 (bits 8,7,6,5,4)
        (@as(u64, 0x1000) << 13) | // diamonds: A (bit 12)
        (@as(u64, 0x0800) << 26); // hearts: K (bit 11)

    try testing.expect(isFlushHand(straight_flush));
    const pattern = getFlushPattern(straight_flush);
    try testing.expectEqual(@as(u16, 0x03E0), pattern); // 9 8 7 6 5 pattern
}

test "single hand evaluation" {
    // Test a specific hand - Royal flush clubs
    const test_hand: u64 = 0x1F00; // A-K-Q-J-T of clubs

    const slow_result = slow_eval.evaluateHand(test_hand);
    const fast_result = evaluateHand(test_hand);

    // Only print on failure
    if (slow_result != fast_result) {
        std.debug.print("Single hand FAIL: hand=0x{X}, slow={}, fast={}\n", .{ test_hand, slow_result, fast_result });
    }

    try testing.expectEqual(slow_result, fast_result);
}

test "batch evaluation" {
    // Generate test batch
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = generateRandomHandBatch(4, &rng);

    // Evaluate batch
    const batch_results = evaluateBatch(4, batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    const batch_size = 4;
    for (0..batch_size) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = slow_eval.evaluateHand(hand);

        if (batch_result == single_result) {
            matches += 1;
        }

        // Only print failures
        if (batch_result != single_result) {
            std.debug.print("Batch FAIL hand {}: batch={}, single={}\n", .{ i, batch_result, single_result });
        }
    }

    // Only print on failure
    if (matches != batch_size) {
        const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(batch_size)) * 100.0;
        std.debug.print("Batch accuracy: {}/{} ({d:.1}%)\n", .{ matches, batch_size, accuracy });
    }

    try testing.expectEqual(@as(u32, @intCast(batch_size)), matches);
}

test "variable batch sizes" {
    var prng = std.Random.DefaultPrng.init(123);
    var rng = prng.random();

    // Test batch sizes 1, 2, 4, 8
    inline for ([_]usize{ 1, 2, 4, 8, 16, 32 }) |batchSize| {
        // Generate random hands
        var hands_array: [batchSize]u64 = undefined;
        for (&hands_array) |*hand| {
            hand.* = generateRandomHand(&rng);
        }
        const hands: @Vector(batchSize, u64) = hands_array;

        // Evaluate using batch function
        const batch_results = evaluateBatch(batchSize, hands);

        // Verify each result
        for (0..batchSize) |i| {
            const expected = slow_eval.evaluateHand(hands_array[i]);
            const actual = batch_results[i];

            if (expected != actual) {
                std.debug.print("Batch size {} FAIL at index {}: expected={}, actual={}\n", .{ batchSize, i, expected, actual });
            }
            try testing.expectEqual(expected, actual);
        }
    }
}

test "two trips makes full house" {
    // Test case: AAAKKK8 - two trips should be a full house
    const test_hand = card.parseCard("Ac") |
        card.parseCard("Ad") |
        card.parseCard("Ah") |
        card.parseCard("Kc") |
        card.parseCard("Kd") |
        card.parseCard("Kh") |
        card.parseCard("8c");

    const slow_rank = slow_eval.evaluateHand(test_hand);
    const fast_rank = evaluateHand(test_hand);

    // Should be a full house
    try testing.expect(slow_rank / slow_eval.CATEGORY_STEP == 2);
    try testing.expect(fast_rank / slow_eval.CATEGORY_STEP == 2);

    // Fast and slow should match
    if (slow_rank != fast_rank) {
        std.debug.print("Two trips FAIL: hand=0x{X}, slow={}, fast={}\n", .{ test_hand, slow_rank, fast_rank });
    }
    try testing.expectEqual(slow_rank, fast_rank);
}

test "two trips edge cases" {
    // Test multiple two-trips scenarios using clear string notation
    const aaakkk8 = card.parseCard("Ac") |
        card.parseCard("Ad") |
        card.parseCard("Ah") |
        card.parseCard("Kc") |
        card.parseCard("Kd") |
        card.parseCard("Kh") |
        card.parseCard("8c");

    const trips_222_333_a = card.parseCard("2c") |
        card.parseCard("2d") |
        card.parseCard("2h") |
        card.parseCard("3c") |
        card.parseCard("3d") |
        card.parseCard("3h") |
        card.parseCard("Ac");

    const trips_qqq_jjj_5 = card.parseCard("Qc") |
        card.parseCard("Qd") |
        card.parseCard("Qh") |
        card.parseCard("Jc") |
        card.parseCard("Jd") |
        card.parseCard("Jh") |
        card.parseCard("5c");

    const trips_777_666_k = card.parseCard("7c") |
        card.parseCard("7d") |
        card.parseCard("7h") |
        card.parseCard("6c") |
        card.parseCard("6d") |
        card.parseCard("6h") |
        card.parseCard("Kc");

    // Test multiple two-trips scenarios
    const test_cases = [_]struct { hand: u64, desc: []const u8 }{
        .{ .hand = aaakkk8, .desc = "AAA KKK 8" },
        .{ .hand = trips_222_333_a, .desc = "222 333 A" },
        .{ .hand = trips_qqq_jjj_5, .desc = "QQQ JJJ 5" },
        .{ .hand = trips_777_666_k, .desc = "777 666 K" },
    };

    for (test_cases) |tc| {
        const slow_rank = slow_eval.evaluateHand(tc.hand);
        const fast_rank = evaluateHand(tc.hand);

        // All should be full houses
        try testing.expect(slow_rank / slow_eval.CATEGORY_STEP == 2);

        if (slow_rank != fast_rank) {
            std.debug.print("Two trips edge case FAIL ({s}): hand=0x{X}, slow={}, fast={}\n", .{ tc.desc, tc.hand, slow_rank, fast_rank });
        }
        try testing.expectEqual(slow_rank, fast_rank);
    }
}

test "verify problem hands from verify-all" {
    // These are actual failing hands from the verify-all output
    const problem_hands = [_]u64{
        0x4802245, // First failing hand
        0x8000802245, // Second failing hand
        0x1004002245, // Third failing hand
        0x4402445, // Another pattern
        0x4C02045, // Another pattern
    };

    for (problem_hands, 0..) |hand, i| {
        const slow_rank = slow_eval.evaluateHand(hand);
        const fast_rank = evaluateHand(hand);
        const category = getHandCategory(fast_rank);

        // Only print debug info if there's a mismatch
        if (slow_rank != fast_rank) {
            std.debug.print("Problem hand {}: 0x{X}\n", .{ i, hand });
            std.debug.print("  Slow rank: {} (category: {})\n", .{ slow_rank, getHandCategory(slow_rank) });
            std.debug.print("  Fast rank: {} (category: {})\n", .{ fast_rank, category });
        }

        if (slow_rank != fast_rank) {
            // Let's decode the hand to understand what cards it has
            std.debug.print("  Hand breakdown:\n", .{});
            for (0..4) |suit| {
                const suit_mask = card.getSuitMask(hand, @enumFromInt(suit));
                if (suit_mask != 0) {
                    std.debug.print("    Suit {}: 0x{X}\n", .{ suit, suit_mask });
                }
            }
        }

        try testing.expectEqual(slow_rank, fast_rank);
    }
}

test "board context showdown matches evaluate" {
    const board = card.makeCard(.hearts, .ace) |
        card.makeCard(.hearts, .king) |
        card.makeCard(.hearts, .queen);
    const hero_hole = card.makeCard(.spades, .jack) | card.makeCard(.clubs, .jack);
    const villain_hole = card.makeCard(.diamonds, .ace) | card.makeCard(.diamonds, .king);

    const ctx = initBoardContext(board);

    const hero_rank_ctx = evaluateHoleWithContext(&ctx, hero_hole);
    const villain_rank_ctx = evaluateHoleWithContext(&ctx, villain_hole);
    const showdown_ctx = evaluateShowdownWithContext(&ctx, hero_hole, villain_hole);

    const hero_final = board | hero_hole;
    const villain_final = board | villain_hole;
    const hero_rank = evaluateHand(hero_final);
    const villain_rank = evaluateHand(villain_final);
    const showdown: i8 = if (hero_rank < villain_rank)
        @as(i8, 1)
    else if (hero_rank > villain_rank)
        @as(i8, -1)
    else
        @as(i8, 0);

    try testing.expectEqual(hero_rank, hero_rank_ctx);
    try testing.expectEqual(villain_rank, villain_rank_ctx);
    try testing.expectEqual(showdown, showdown_ctx);
}

test "evaluateShowdownMultiway identifies unique winner" {
    const board = card.makeCard(.spades, .ace) |
        card.makeCard(.spades, .king) |
        card.makeCard(.spades, .queen) |
        card.makeCard(.spades, .jack) |
        card.makeCard(.hearts, .two);

    const ctx = initBoardContext(board);
    const holes = [_]card.Hand{
        card.makeCard(.spades, .ten) | card.makeCard(.clubs, .three),
        card.makeCard(.hearts, .ace) | card.makeCard(.diamonds, .ace),
        card.makeCard(.clubs, .king) | card.makeCard(.clubs, .queen),
    };

    const result = evaluateShowdownMultiway(&ctx, &holes);
    try testing.expectEqual(@as(u8, 1), result.tie_count);
    try testing.expectEqual(@as(u64, 1), result.winner_mask);

    const best_rank = evaluateHand(board | holes[0]);
    try testing.expectEqual(best_rank, result.best_rank);
}

test "evaluateEquityWeights splits pot evenly" {
    const board = card.makeCard(.spades, .ace) |
        card.makeCard(.spades, .king) |
        card.makeCard(.spades, .queen) |
        card.makeCard(.spades, .jack) |
        card.makeCard(.spades, .ten);

    const ctx = initBoardContext(board);
    const holes = [_]card.Hand{
        card.makeCard(.hearts, .two) | card.makeCard(.diamonds, .three),
        card.makeCard(.clubs, .four) | card.makeCard(.diamonds, .five),
        card.makeCard(.hearts, .six) | card.makeCard(.diamonds, .seven),
    };

    var equities = [_]f64{ 0.0, 0.0, 0.0 };
    const result = evaluateEquityWeights(&ctx, &holes, &equities);

    try testing.expectEqual(@as(u8, 3), result.tie_count);
    try testing.expectEqual(@as(u64, 0b111), result.winner_mask);

    const share = 1.0 / 3.0;
    try testing.expectApproxEqAbs(share, equities[0], 1e-9);
    try testing.expectApproxEqAbs(share, equities[1], 1e-9);
    try testing.expectApproxEqAbs(share, equities[2], 1e-9);
}

test "board context batch showdown matches single" {
    const PairCount = 37;
    const board = card.makeCard(.hearts, .ace) |
        card.makeCard(.hearts, .king) |
        card.makeCard(.hearts, .queen) |
        card.makeCard(.diamonds, .two) |
        card.makeCard(.clubs, .three);
    const ctx = initBoardContext(board);

    var hero_holes: [PairCount]card.Hand = undefined;
    var villain_holes: [PairCount]card.Hand = undefined;

    const Picker = struct {
        fn pick(base_suit: card.Suit, base_rank_hint: usize, used: *card.Hand) card.Hand {
            var suit = base_suit;
            var rank_index: usize = base_rank_hint % 13;
            var attempts: usize = 0;
            while (attempts < 52) : (attempts += 1) {
                const rank: card.Rank = @enumFromInt(rank_index % 13);
                const card_bit = card.makeCard(suit, rank);
                if ((used.* & card_bit) == 0) {
                    used.* |= card_bit;
                    return card_bit;
                }
                rank_index = (rank_index + 3) % 13;
                suit = switch (suit) {
                    .clubs => .diamonds,
                    .diamonds => .spades,
                    .spades => .hearts,
                    .hearts => .clubs,
                };
            }
            @panic("unable to pick distinct card");
        }
    };

    var idx: usize = 0;
    while (idx < PairCount) : (idx += 1) {
        var used = board;
        const hero_card1 = Picker.pick(.clubs, idx * 2, &used);
        const hero_card2 = Picker.pick(.diamonds, idx * 2 + 5, &used);
        const hero_hole = hero_card1 | hero_card2;

        const villain_card1 = Picker.pick(.spades, idx * 3 + 2, &used);
        const villain_card2 = Picker.pick(.hearts, idx * 5 + 7, &used);
        const villain_hole = villain_card1 | villain_card2;
        std.debug.assert((hero_hole & villain_hole) == 0);

        hero_holes[idx] = hero_hole;
        villain_holes[idx] = villain_hole;
    }

    var results = [_]i8{0} ** PairCount;
    evaluateShowdownBatch(&ctx, hero_holes[0..], villain_holes[0..], results[0..]);

    for (hero_holes, 0..) |hero_hole, j| {
        const expected = evaluateShowdownWithContext(&ctx, hero_hole, villain_holes[j]);
        try testing.expectEqual(expected, results[j]);
    }
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
