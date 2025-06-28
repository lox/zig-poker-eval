const std = @import("std");
const tables = @import("tables.zig");

// High-performance evaluator with SIMD batching
const RANK_MASK = 0x1FFF; // 13 bits for ranks

// === Scalar RPC Computation (for single hands and flushes) ===

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

// === SIMD RPC Computation (for 4-hand batches) ===

fn computeRpcSimd4(hands: [4]u64) [4]u32 {
    // Extract suits for all 4 hands (structure-of-arrays)
    var clubs: [4]u16 = undefined;
    var diamonds: [4]u16 = undefined; 
    var hearts: [4]u16 = undefined;
    var spades: [4]u16 = undefined;
    
    for (hands, 0..) |hand, i| {
        clubs[i] = @as(u16, @truncate((hand >> 0) & RANK_MASK));
        diamonds[i] = @as(u16, @truncate((hand >> 13) & RANK_MASK));
        hearts[i] = @as(u16, @truncate((hand >> 26) & RANK_MASK));
        spades[i] = @as(u16, @truncate((hand >> 39) & RANK_MASK));
    }
    
    const clubs_v: @Vector(4, u16) = clubs;
    const diamonds_v: @Vector(4, u16) = diamonds;
    const hearts_v: @Vector(4, u16) = hearts;
    const spades_v: @Vector(4, u16) = spades;
    
    var rpc_vec: @Vector(4, u32) = @splat(0);
    
    // Vectorized rank counting for all 13 ranks
    inline for (0..13) |rank| {
        const rank_bit: @Vector(4, u16) = @splat(@as(u16, 1) << @intCast(rank));
        const zero_vec: @Vector(4, u16) = @splat(0);
        
        // Count rank occurrences across all suits (vectorized)
        const one_vec: @Vector(4, u8) = @splat(1);
        const zero_u8_vec: @Vector(4, u8) = @splat(0);
        
        const clubs_has = @select(u8, (clubs_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const diamonds_has = @select(u8, (diamonds_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const hearts_has = @select(u8, (hearts_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const spades_has = @select(u8, (spades_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        
        // Sum to get rank count for each hand
        const rank_count_vec = clubs_has + diamonds_has + hearts_has + spades_has;
        
        // Vectorized base-5 encoding: rpc = rpc * 5 + count
        const five_vec: @Vector(4, u32) = @splat(5);
        rpc_vec = rpc_vec * five_vec + @as(@Vector(4, u32), rank_count_vec);
    }
    
    return @as([4]u32, rpc_vec);
}


fn chdLookupScalar(rpc: u32) u16 {
    return tables.lookup(rpc);
}

pub fn isFlushHand(hand: u64) bool {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK,   // clubs
        @as(u16, @truncate(hand >> 13)) & RANK_MASK,  // diamonds
        @as(u16, @truncate(hand >> 26)) & RANK_MASK,  // hearts
        @as(u16, @truncate(hand >> 39)) & RANK_MASK,  // spades
    };
    
    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) return true;
    }
    return false;
}

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

pub fn evaluateHand(hand: u64) u16 {
    if (isFlushHand(hand)) {
        const pattern = getFlushPattern(hand);
        return tables.flushLookup(pattern);
    }
    
    const rpc = computeRpcFromHand(hand);
    return chdLookupScalar(rpc);
}


// Simple batch evaluation - leverages SIMD RPC computation with clean control flow
// 
// This approach was chosen over complex flush mask logic because:
// - Same performance (7.24 ns/hand) as the complex version (7.29 ns/hand)
// - Much more readable (8 lines vs 25 lines)
// - Better LLVM IR generation (more vectorization, no memory allocation)
// - Simpler control flow allows compiler optimizations to work effectively
//
// The SIMD benefits come from compute_rpc_simd4(), not from manual vectorization
// of the control flow, so we let the compiler handle the conditional evaluation.
pub fn evaluateBatch4(hands: @Vector(4, u64)) @Vector(4, u16) {
    const hands_array = [4]u64{ hands[0], hands[1], hands[2], hands[3] };
    
    // Compute RPC for all 4 hands using SIMD (this is where the real speedup comes from)
    const rpc_results = computeRpcSimd4(hands_array);
    
    // Simple per-hand evaluation - compiler vectorizes this effectively
    return @Vector(4, u16){
        if (isFlushHand(hands_array[0])) tables.flushLookup(getFlushPattern(hands_array[0])) else chdLookupScalar(rpc_results[0]),
        if (isFlushHand(hands_array[1])) tables.flushLookup(getFlushPattern(hands_array[1])) else chdLookupScalar(rpc_results[1]),
        if (isFlushHand(hands_array[2])) tables.flushLookup(getFlushPattern(hands_array[2])) else chdLookupScalar(rpc_results[2]),
        if (isFlushHand(hands_array[3])) tables.flushLookup(getFlushPattern(hands_array[3])) else chdLookupScalar(rpc_results[3]),
    };
}

// Architecture-adaptive batch processing - processes hands in 4-hand SIMD batches
pub fn evaluateBatchDynamic(hands: []const u64, results: []u16) void {
    std.debug.assert(hands.len == results.len);
    
    // Process in chunks of 4 (optimal for SIMD)
    var i: usize = 0;
    while (i + 4 <= hands.len) : (i += 4) {
        const batch_hands = @Vector(4, u64){ hands[i], hands[i+1], hands[i+2], hands[i+3] };
        const batch_results = evaluateBatch4(batch_hands);
        
        results[i] = batch_results[0];
        results[i+1] = batch_results[1];
        results[i+2] = batch_results[2];
        results[i+3] = batch_results[3];
    }
    
    // Handle remainder
    while (i < hands.len) : (i += 1) {
        results[i] = evaluateHand(hands[i]);
    }
}

// === Benchmarking ===

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
    const test_hands = @Vector(4, u64){ 
        0x1F00000000000, 0x123456789ABCD, 0x0F0F0F0F0F0F0, 0x1F00 
    };
    
    for (0..iterations) |_| {
        const results = evaluateBatch4(test_hands);
        sum +%= results[0] + results[1] + results[2] + results[3];
    }
    
    return sum;
}

// === Test Utilities ===

const slow_evaluator = @import("slow_evaluator.zig");

// Generate random hands for testing (4-hand batches)
pub fn generateRandomHandBatch(rng: *std.Random) @Vector(4, u64) {
    var hands: [4]u64 = undefined;

    for (&hands) |*hand| {
        hand.* = generateRandomHand(rng);
    }

    return @as(@Vector(4, u64), hands);
}

pub fn generateRandomHand(rng: *std.Random) u64 {
    var hand: u64 = 0;
    var cards_dealt: u8 = 0;

    while (cards_dealt < 7) {
        const suit = rng.intRangeAtMost(u8, 0, 3);
        const rank = rng.intRangeAtMost(u8, 0, 12);
        const card = slow_evaluator.makeCard(suit, rank);

        if ((hand & card) == 0) {
            hand |= card;
            cards_dealt += 1;
        }
    }

    return hand;
}

// === Tests ===

test "flush pattern extraction" {
    // Test royal flush in spades: As Ks Qs Js Ts + 2 non-spade cards
    const royal_flush: u64 = 
        (@as(u64, 0x1F00) << 39) | // spades: A K Q J T (bits 12,11,10,9,8)
        (@as(u64, 0x0040) << 26) | // hearts: 7 (bit 6) 
        (@as(u64, 0x0020) << 13);  // diamonds: 6 (bit 5)
    
    try std.testing.expect(isFlushHand(royal_flush));
    const pattern = getFlushPattern(royal_flush);
    try std.testing.expectEqual(@as(u16, 0x1F00), pattern); // A K Q J T pattern
}

test "straight flush pattern" {
    // Test straight flush 9-5 in clubs: 9c 8c 7c 6c 5c + 2 non-club cards
    const straight_flush: u64 = 
        (@as(u64, 0x03E0) << 0) |  // clubs: 9 8 7 6 5 (bits 8,7,6,5,4)
        (@as(u64, 0x1000) << 13) | // diamonds: A (bit 12)
        (@as(u64, 0x0800) << 26);  // hearts: K (bit 11)
    
    try std.testing.expect(isFlushHand(straight_flush));
    const pattern = getFlushPattern(straight_flush);
    try std.testing.expectEqual(@as(u16, 0x03E0), pattern); // 9 8 7 6 5 pattern
}

test "single hand evaluation" {
    // Test a specific hand - Royal flush clubs
    const test_hand: u64 = 0x1F00; // A-K-Q-J-T of clubs

    const slow_result = slow_evaluator.evaluateHand(test_hand);
    const fast_result = evaluateHand(test_hand);

    // Only print on failure
    if (slow_result != fast_result) {
        std.debug.print("Single hand FAIL: hand=0x{X}, slow={}, fast={}\n", .{ test_hand, slow_result, fast_result });
    }

    try std.testing.expectEqual(slow_result, fast_result);
}

test "batch evaluation" {
    // Generate test batch
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = generateRandomHandBatch(&rng);

    // Evaluate batch
    const batch_results = evaluateBatch4(batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    const batch_size = 4;
    for (0..batch_size) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = slow_evaluator.evaluateHand(hand);

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
    
    try std.testing.expectEqual(@as(u32, @intCast(batch_size)), matches);
}

