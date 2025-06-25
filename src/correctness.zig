const std = @import("std");
const evaluator = @import("evaluator.zig");
const simd_evaluator = @import("simd_evaluator.zig");

// Poker hand correctness verifier - catches ranking bugs that tests miss
// Validates known hands against expected poker rankings

const TestHand = struct {
    name: []const u8,
    cards: [7]struct { suit: u8, rank: u8 },
    expected_category: u16, // Hand strength value from evaluator
    is_flush: bool,
    description: []const u8,
};

// Known test hands with expected results
const TEST_HANDS = [_]TestHand{
    // Flush hands (these will expose the BBHash bug)
    .{
        .name = "Royal Flush",
        .cards = .{
            .{ .suit = 0, .rank = 12 }, // Ace of spades
            .{ .suit = 0, .rank = 11 }, // King of spades  
            .{ .suit = 0, .rank = 10 }, // Queen of spades
            .{ .suit = 0, .rank = 9 },  // Jack of spades
            .{ .suit = 0, .rank = 8 },  // 10 of spades
            .{ .suit = 1, .rank = 7 },  // 9 of hearts (extra)
            .{ .suit = 2, .rank = 6 },  // 8 of diamonds (extra)
        },
        .expected_category = 7461, // Royal flush value
        .is_flush = true,
        .description = "A-K-Q-J-10 all spades (best possible hand)",
    },
    .{
        .name = "Ace High Flush",
        .cards = .{
            .{ .suit = 1, .rank = 12 }, // Ace of hearts
            .{ .suit = 1, .rank = 10 }, // Jack of hearts
            .{ .suit = 1, .rank = 8 },  // 9 of hearts
            .{ .suit = 1, .rank = 6 },  // 7 of hearts
            .{ .suit = 1, .rank = 4 },  // 5 of hearts
            .{ .suit = 0, .rank = 3 },  // 4 of spades (extra)
            .{ .suit = 2, .rank = 2 },  // 3 of diamonds (extra)
        },
        .expected_category = 5000, // Flush
        .is_flush = true,
        .description = "A-J-9-7-5 all hearts",
    },
    
    // Non-flush hands (these should work with CHD)
    .{
        .name = "Four Aces",
        .cards = .{
            .{ .suit = 0, .rank = 12 }, // Ace of spades
            .{ .suit = 1, .rank = 12 }, // Ace of hearts
            .{ .suit = 2, .rank = 12 }, // Ace of diamonds
            .{ .suit = 3, .rank = 12 }, // Ace of clubs
            .{ .suit = 0, .rank = 11 }, // King of spades
            .{ .suit = 1, .rank = 10 }, // Queen of hearts
            .{ .suit = 2, .rank = 9 },  // Jack of diamonds
        },
        .expected_category = 7000, // Four of a kind
        .is_flush = false,
        .description = "Four aces with K-Q-J",
    },
    .{
        .name = "Full House",
        .cards = .{
            .{ .suit = 0, .rank = 11 }, // King of spades
            .{ .suit = 1, .rank = 11 }, // King of hearts
            .{ .suit = 2, .rank = 11 }, // King of diamonds
            .{ .suit = 0, .rank = 10 }, // Queen of spades
            .{ .suit = 1, .rank = 10 }, // Queen of hearts
            .{ .suit = 2, .rank = 9 },  // Jack of diamonds
            .{ .suit = 3, .rank = 8 },  // 10 of clubs
        },
        .expected_category = 6000, // Full house
        .is_flush = false,
        .description = "Kings full of Queens",
    },
    .{
        .name = "Non-Flush Straight",
        .cards = .{
            .{ .suit = 0, .rank = 8 },  // 9 of spades
            .{ .suit = 1, .rank = 7 },  // 8 of hearts
            .{ .suit = 2, .rank = 6 },  // 7 of diamonds
            .{ .suit = 3, .rank = 5 },  // 6 of clubs
            .{ .suit = 0, .rank = 4 },  // 5 of spades
            .{ .suit = 1, .rank = 3 },  // 4 of hearts
            .{ .suit = 2, .rank = 2 },  // 3 of diamonds
        },
        .expected_category = 4000, // Straight
        .is_flush = false,
        .description = "9-high straight (mixed suits)",
    },
    .{
        .name = "High Card Only",
        .cards = .{
            .{ .suit = 0, .rank = 12 }, // Ace of spades
            .{ .suit = 1, .rank = 10 }, // Jack of hearts
            .{ .suit = 2, .rank = 8 },  // 9 of diamonds
            .{ .suit = 3, .rank = 6 },  // 7 of clubs
            .{ .suit = 0, .rank = 4 },  // 5 of spades
            .{ .suit = 1, .rank = 2 },  // 3 of hearts
            .{ .suit = 2, .rank = 0 },  // 2 of diamonds
        },
        .expected_category = 0, // High card
        .is_flush = false,
        .description = "Ace high, no pairs or flushes",
    },
};

// Convert test hand to evaluator hand
fn testHandToHand(test_hand: TestHand) evaluator.Hand {
    var hand: evaluator.Hand = 0;
    
    for (test_hand.cards) |card| {
        hand |= evaluator.makeCard(card.suit, card.rank);
    }
    
    return hand;
}

// Run comprehensive correctness verification
pub fn verifyCorrectness() !void {
    const print = std.debug.print;
    
    print("Poker Hand Correctness Verification\n", .{});
    print("===================================\n", .{});
    
    var total_tests: u32 = 0;
    var passing_tests: u32 = 0;
    var flush_tests: u32 = 0;
    var flush_bugs: u32 = 0;
    
    const simd_eval = simd_evaluator.SimdEvaluator.init();
    
    for (TEST_HANDS) |test_hand| {
        total_tests += 1;
        
        print("\nTesting: {s}\n", .{test_hand.name});
        print("  Description: {s}\n", .{test_hand.description});
        print("  Expected category: {} ({s})\n", .{ test_hand.expected_category, getCategoryName(test_hand.expected_category) });
        print("  Is flush: {}\n", .{test_hand.is_flush});
        
        const hand = testHandToHand(test_hand);
        
        // Test reference evaluator
        const ref_result = evaluator.evaluateHand(hand);
        const ref_category = evaluator.getHandCategory(hand);
        
        print("  Reference result: {} (category {})\n", .{ ref_result, ref_category });
        
        // Test SIMD evaluator (single hand in batch)
        var batch: [16]u64 = std.mem.zeroes([16]u64);
        batch[0] = hand;
        const simd_batch = simd_evaluator.HandBatch{ batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7], batch[8], batch[9], batch[10], batch[11], batch[12], batch[13], batch[14], batch[15] };
        const simd_results = simd_eval.evaluateBatch(simd_batch);
        const simd_result = simd_results[0];
        
        print("  SIMD result: {}\n", .{simd_result});
        
        // Debug non-flush hands in detail
        if (!test_hand.is_flush) {
            const chd = @import("chd.zig");
            const rank_counts = chd.getRankCounts(hand);
            const rpc = chd.encodeRPC(rank_counts);
            
            print("  DEBUG: Hand=0x{x:013}\n", .{hand});
            print("  DEBUG: RPC=0x{x:08}\n", .{rpc});
            print("  DEBUG: Rank counts: ", .{});
            for (rank_counts, 0..) |count, i| {
                if (count > 0) print("{}:{} ", .{i, count});
            }
            print("\n", .{});
        }
        
        // Verify category correctness
        const category_match = ref_category == test_hand.expected_category;
        if (category_match) {
            print("  ✅ Category correct\n", .{});
            passing_tests += 1;
        } else {
            print("  ❌ Category mismatch! Expected {}, got {}\n", .{ test_hand.expected_category, ref_category });
        }
        
        // Check for flush handling bugs
        if (test_hand.is_flush) {
            flush_tests += 1;
            
            // SIMD should return reasonable flush ranking, not just rank mask
            if (simd_result < 8192 and simd_result != ref_result) {
                print("  ⚠️  FLUSH BUG: SIMD result {} doesn't match reference {}\n", .{ simd_result, ref_result });
                print("      This likely indicates placeholder BBHash implementation\n", .{});
                flush_bugs += 1;
            }
        }
        
        // Consistency check between evaluators
        const results_match = simd_result == ref_result;
        if (!results_match and !test_hand.is_flush) {
            print("  ⚠️  NON-FLUSH INCONSISTENCY: SIMD {} vs Reference {}\n", .{ simd_result, ref_result });
        }
    }
    
    // Summary
    print("\n==================================================\n", .{});
    print("CORRECTNESS VERIFICATION SUMMARY\n", .{});
    print("==================================================\n", .{});
    print("Total tests: {}\n", .{total_tests});
    print("Passing tests: {}\n", .{passing_tests});
    print("Success rate: {d:.1}%\n", .{@as(f64, @floatFromInt(passing_tests)) * 100.0 / @as(f64, @floatFromInt(total_tests))});
    
    if (flush_tests > 0) {
        print("\nFLUSH HANDLING:\n", .{});
        print("Flush tests: {}\n", .{flush_tests});
        print("Flush bugs detected: {}\n", .{flush_bugs});
        
        if (flush_bugs > 0) {
            print("❌ FLUSH BUGS DETECTED - BBHash implementation needed\n", .{});
        } else {
            print("✅ Flush handling appears correct\n", .{});
        }
    }
    
    if (passing_tests == total_tests and flush_bugs == 0) {
        print("\n🎯 ALL TESTS PASSED - Evaluator is correct!\n", .{});
    } else {
        print("\n⚠️  ISSUES DETECTED - See details above\n", .{});
    }
}

fn getCategoryName(category: u16) []const u8 {
    return switch (category) {
        0 => "High Card",
        1000 => "Pair", 
        2000 => "Two Pair",
        3000 => "Three of a Kind",
        4000 => "Straight",
        5000 => "Flush",
        6000 => "Full House", 
        7000 => "Four of a Kind",
        7461 => "Royal Flush",
        else => "Other",
    };
}

// Test integration
test "poker correctness verification" {
    try verifyCorrectness();
}
