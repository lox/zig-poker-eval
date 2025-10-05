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
    // Evaluator uses lower numbers for better hands
    // These ranges are based on the actual slow evaluator implementation
    if (rank <= 10) return .straight_flush; // Royal flush + straight flushes (0-10)
    if (rank <= 165) return .four_of_a_kind; // Four of a kind (10-165)
    if (rank <= 321) return .full_house; // Full house (166-321)
    if (rank <= 1598) return .flush; // Flush (322-1598)
    if (rank <= 1608) return .straight; // Straight (1599-1608)
    if (rank <= 2466) return .three_of_a_kind; // Three of a kind (1609-2466)
    if (rank <= 3324) return .two_pair; // Two pair (2467-3324)
    if (rank <= 6184) return .pair; // One pair (3325-6184)
    return .high_card; // High card (6185-7461)
}

// === Core Constants ===

const RANK_MASK = 0x1FFF; // 13 bits for ranks

/// Cached board analysis for reusing evaluation work across multiple hole cards
pub const BoardContext = struct {
    board: card.Hand,
    suit_masks: [4]u16,
    suit_counts: [4]u8,
    rank_counts: [13]u8,
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

    return .{
        .board = board,
        .suit_masks = suit_masks,
        .suit_counts = suit_counts,
        .rank_counts = rank_counts,
    };
}

fn evaluateHoleWithContextImpl(ctx: *const BoardContext, hole: card.Hand) HandRank {
    std.debug.assert((hole & ctx.board) == 0);

    var suit_masks = ctx.suit_masks;
    var suit_counts = ctx.suit_counts;
    var rank_counts = ctx.rank_counts;

    var remaining = hole;
    while (remaining != 0) {
        const bit_index: u6 = @intCast(@ctz(remaining));
        remaining &= remaining - 1;
        const suit_index: usize = @intCast(bit_index / 13);
        const rank_index: usize = @intCast(bit_index % 13);

        suit_masks[suit_index] |= @as(u16, 1) << @intCast(rank_index);
        suit_counts[suit_index] += 1;
        rank_counts[rank_index] += 1;
    }

    inline for (0..4) |suit| {
        if (suit_counts[suit] >= 5) {
            const pattern = getTop5Ranks(suit_masks[suit]);
            return tables.flushLookup(pattern);
        }
    }

    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return tables.lookup(rpc);
}

/// Evaluate a pair of hole cards using a precomputed board context
pub fn evaluateHoleWithContext(ctx: *const BoardContext, hole: card.Hand) HandRank {
    return evaluateHoleWithContextImpl(ctx, hole);
}

/// Evaluate a showdown using shared board context
pub fn evaluateShowdownWithContext(ctx: *const BoardContext, hero_hole: card.Hand, villain_hole: card.Hand) i8 {
    std.debug.assert((hero_hole & villain_hole) == 0);
    std.debug.assert((hero_hole & ctx.board) == 0);
    std.debug.assert((villain_hole & ctx.board) == 0);

    const hero_rank = evaluateHoleWithContextImpl(ctx, hero_hole);
    const villain_rank = evaluateHoleWithContextImpl(ctx, villain_hole);
    return if (hero_rank < villain_rank) 1 else if (hero_rank > villain_rank) -1 else 0;
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

    // TODO(zig-0.15.1): SIMD has a compiler bug on x86_64 Linux
    // error(x86_64_encoder): no encoding found for: none vpmovzxbd ymm m128 none none
    // Disable SIMD on x86_64 until this is fixed
    const builtin = @import("builtin");
    const use_simd = builtin.cpu.arch != .x86_64;

    // Use SIMD for batch sizes that are powers of 2 or have good SIMD support
    if (use_simd and (batchSize == 2 or batchSize == 4 or batchSize == 8 or batchSize == 16 or batchSize == 32 or batchSize == 64)) {
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

fn chdLookupScalar(rpc: u32) u16 {
    return tables.lookup(rpc);
}

// === Flush Detection and Pattern Extraction ===

pub fn isFlushHand(hand: u64) bool {
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

fn getTop5Ranks(suit_mask: u16) u16 {
    if (@popCount(suit_mask) == 5) return suit_mask;

    // Check for straights first
    const straights = [_]u16{
        0x1F00, 0x0F80, 0x07C0, 0x03E0, 0x01F0,
        0x00F8, 0x007C, 0x003E, 0x001F, 0x100F,
    };

    for (straights) |pattern| {
        if ((suit_mask & pattern) == pattern) return pattern;
    }

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

    return result;
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

    // Compute RPC for all hands (SIMD-optimized for batchSize=4)
    const rpc_results = computeRpcSimd(batchSize, &hands_array);

    // Evaluate each hand
    var results: [batchSize]u16 = undefined;
    inline for (0..batchSize) |i| {
        if (isFlushHand(hands_array[i])) {
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
    try testing.expect(getHandCategory(1) == .straight_flush);
    try testing.expect(getHandCategory(100) == .four_of_a_kind);
    try testing.expect(getHandCategory(7000) == .high_card);
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
    inline for ([_]usize{ 1, 2, 4, 8 }) |batchSize| {
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

    // Should be a full house (rank 166-321)
    try testing.expect(slow_rank >= 166 and slow_rank <= 321);
    try testing.expect(fast_rank >= 166 and fast_rank <= 321);

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

        // All should be full houses (rank 166-321)
        try testing.expect(slow_rank >= 166 and slow_rank <= 321);

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
