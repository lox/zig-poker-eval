const std = @import("std");
const simd_evaluator = @import("simd_evaluator.zig");
const slow_evaluator = @import("slow_evaluator.zig");

// Generate random hands for testing (consolidated from bench.zig)
pub fn generateRandomHandBatch(rng: *std.Random) simd_evaluator.VecU64 {
    var hands: [16]u64 = undefined;

    for (&hands) |*hand| {
        hand.* = generateRandomHand(rng);
    }

    return simd_evaluator.VecU64{ hands[0], hands[1], hands[2], hands[3], hands[4], hands[5], hands[6], hands[7], hands[8], hands[9], hands[10], hands[11], hands[12], hands[13], hands[14], hands[15] };
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

// Tests that run with `zig build test`

test "known hand types" {

    const test_hands = [_]struct {
        hand: u64,
        name: []const u8,
        expected_rank_min: u16,
        expected_rank_max: u16,
    }{
        // Royal flush clubs
        .{ .hand = 0x1F00 | (1 << 13) | (1 << 26), .name = "Royal Flush", .expected_rank_min = 0, .expected_rank_max = 9 },

        // Wheel straight flush (A-2-3-4-5) clubs + 2 off-suit cards  
        // Fixed: Now correctly detects as straight flush (rank 9)
        .{ .hand = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << (13 + 10)) | (1 << (26 + 8)), .name = "Wheel Straight Flush", .expected_rank_min = 9, .expected_rank_max = 9 },

        // Four aces
        .{ .hand = (1 << 12) | (1 << (13 + 12)) | (1 << (26 + 12)) | (1 << (39 + 12)) | (1 << 11) | (1 << (13 + 10)) | (1 << (26 + 9)), .name = "Four Aces", .expected_rank_min = 10, .expected_rank_max = 165 },

        // Full house
        .{ .hand = (1 << 10) | (1 << (13 + 10)) | (1 << (26 + 10)) | (1 << 9) | (1 << (13 + 9)) | (1 << (26 + 8)) | (1 << (39 + 7)), .name = "Full House", .expected_rank_min = 166, .expected_rank_max = 321 },

        // Straight
        .{ .hand = (1 << 8) | (1 << (13 + 7)) | (1 << (26 + 6)) | (1 << (39 + 5)) | (1 << 4) | (1 << (13 + 2)) | (1 << (26 + 0)), .name = "Straight", .expected_rank_min = 1599, .expected_rank_max = 1608 },

        // High card
        .{ .hand = (1 << 12) | (1 << (13 + 10)) | (1 << (26 + 8)) | (1 << (39 + 6)) | (1 << 4) | (1 << (13 + 2)) | (1 << (26 + 0)), .name = "High Card", .expected_rank_min = 6185, .expected_rank_max = 7461 },
    };

    var all_match = true;

    for (test_hands) |test_case| {
        const slow_rank = slow_evaluator.evaluateHand(test_case.hand);
        const fast_rank = simd_evaluator.evaluate_single_hand(test_case.hand);

        const match = slow_rank == fast_rank;

        // Only print on failure
        if (!match) {
            std.debug.print("FAIL {s}: slow={}, fast={}\n", .{ test_case.name, slow_rank, fast_rank });
        }

        if (!match) all_match = false;
    }

    // Only print summary on failure
    if (!all_match) {
        std.debug.print("Known hand types: some tests failed\n", .{});
    }
    try std.testing.expect(all_match);
}

test "random hands validation" {
    const num_hands = 10000; // Reduced for faster tests

    var prng = std.Random.DefaultPrng.init(0x12345678);
    var rand = prng.random();

    var correct_count: u32 = 0;
    var total_count: u32 = 0;
    var mismatches: u32 = 0;

    for (0..num_hands) |_| {
        // Generate a random 7-card hand by selecting 7 cards from 52
        var hand: u64 = 0;
        var cards_selected: u8 = 0;
        var attempts: u16 = 0;

        while (cards_selected < 7 and attempts < 1000) {
            attempts += 1;
            const card_idx = rand.intRangeAtMost(u8, 0, 51);
            const suit = card_idx / 13;
            const rank = card_idx % 13;
            const card_bit = slow_evaluator.makeCard(suit, rank);

            // Only add if this card isn't already in the hand
            if ((hand & card_bit) == 0) {
                hand |= card_bit;
                cards_selected += 1;
            }
        }

        if (cards_selected == 7) {
            const slow_rank = slow_evaluator.evaluateHand(hand);
            const fast_rank = simd_evaluator.evaluate_single_hand(hand);

            total_count += 1;

            if (slow_rank == fast_rank) {
                correct_count += 1;
            } else {
                mismatches += 1;
                if (mismatches <= 3) { // Show first 3 mismatches for debugging
                    std.debug.print("MISMATCH #{}: hand=0x{X}, slow={}, fast={}\n", .{ mismatches, hand, slow_rank, fast_rank });
                }
            }
        }
    }

    const accuracy = @as(f64, @floatFromInt(correct_count)) / @as(f64, @floatFromInt(total_count)) * 100.0;
    
    // Only print on failure or low accuracy
    if (accuracy < 100.0) {
        std.debug.print("Random hands: {}/{} correct ({d:.2}%), mismatches={}\n", .{ correct_count, total_count, accuracy, mismatches });
    }
    
    // Enforce 100% accuracy requirement
    try std.testing.expectEqual(@as(f64, 100.0), accuracy);
}

test "single hand evaluation" {

    // Test a specific hand - Royal flush clubs
    const test_hand: u64 = 0x1F00; // A-K-Q-J-T of clubs

    const slow_result = slow_evaluator.evaluateHand(test_hand);
    const fast_result = simd_evaluator.evaluate_single_hand(test_hand);

    // Only print on failure
    if (slow_result != fast_result) {
        std.debug.print("Single hand FAIL: hand=0x{X}, slow={}, fast={}\n", .{ test_hand, slow_result, fast_result });
    }

    try std.testing.expectEqual(slow_result, fast_result);
}

test "SIMD batch evaluation" {

    const simd_eval = simd_evaluator.SIMDEvaluator.init();

    // Generate test batch
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = generateRandomHandBatch(&rng);

    // Evaluate batch
    const batch_results = simd_eval.evaluate_batch(batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    for (0..16) |i| {
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
    if (matches != 16) {
        const accuracy = @as(f64, @floatFromInt(matches)) / 16.0 * 100.0;
        std.debug.print("Batch accuracy: {}/16 ({d:.1}%)\n", .{ matches, accuracy });
    }
    
    try std.testing.expectEqual(@as(u32, 16), matches);
}

test "batch correctness validation" {

    const simd_eval = simd_evaluator.SIMDEvaluator.init();
    var prng = std.Random.DefaultPrng.init(0x12345678);
    var rng = prng.random();

    // Generate test batches (reduced for faster tests)
    var test_batches: [100]simd_evaluator.VecU64 = undefined;
    for (&test_batches) |*batch| {
        batch.* = generateRandomHandBatch(&rng);
    }

    var matches: u32 = 0;
    var total: u32 = 0;

    // Validate batches (16K hands)
    for (test_batches) |batch| {
        const fast_results = simd_eval.evaluate_batch(batch);

        for (0..16) |j| {
            const slow_result = slow_evaluator.evaluateHand(batch[j]);
            const fast_result = fast_results[j];

            if (slow_result == fast_result) {
                matches += 1;
            }
            total += 1;
        }
    }

    const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total)) * 100.0;
    
    // Only print on failure
    if (accuracy < 100.0) {
        std.debug.print("Batch validation: {}/{} correct ({d:.2}%)\n", .{ matches, total, accuracy });
    }

    try std.testing.expectEqual(@as(f64, 100.0), accuracy);
}
