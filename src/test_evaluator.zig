const std = @import("std");
const testing = std.testing;
const evaluator = @import("evaluator.zig");

// Helper function to create a card: rank (0-12 for 2-A), suit (0-3 for ♠♥♦♣)
fn makeCard(rank: u4, suit: u2) u8 {
    return (@as(u8, rank) << 2) | suit;
}

// Helper function to evaluate a hand and return strength
fn evalHand(cards: []const u8) u16 {
    const bits = evaluator.encodeCards(cards);
    return evaluator.eval7(bits);
}

test "royal flush detection" {
    std.debug.print("🎰 Testing royal flush detection...\n", .{});
    // Royal flush: As Ks Qs Js 10s + extra cards in spades
    const royal_hand = [_]u8{ 
        (12<<2)|0, (11<<2)|0, (10<<2)|0, (9<<2)|0, (8<<2)|0, (7<<2)|0, (6<<2)|0
    };
    
    const strength = evalHand(&royal_hand);
    std.debug.print("  Royal flush strength: {}\n", .{strength});
    try testing.expect(strength >= 9000); // Royal flush
    try testing.expectEqualStrings("Royal Flush", evaluator.strengthToString(strength));
    std.debug.print("  ✅ Royal flush test passed\n", .{});
}

test "four of a kind detection" {
    std.debug.print("🃏 Testing four of a kind detection...\n", .{});
    // Four aces + random cards
    const quads = [_]u8{
        makeCard(12, 0), makeCard(12, 1), makeCard(12, 2), makeCard(12, 3), // Four aces
        makeCard(10, 0), makeCard(9, 1), makeCard(8, 2) // Random cards
    };
    
    const strength = evalHand(&quads);
    std.debug.print("  Four of a kind strength: {}\n", .{strength});
    try testing.expect(strength >= 7000 and strength < 8000); // Four of a kind
    try testing.expectEqualStrings("Four of a Kind", evaluator.strengthToString(strength));
    std.debug.print("  ✅ Four of a kind test passed\n", .{});
}

test "basic hand type ranges" {
    std.debug.print("📊 Testing hand type ranges...\n", .{});
    // Test that hand types fall within expected ranges
    const royal = [_]u8{ (12<<2)|0, (11<<2)|0, (10<<2)|0, (9<<2)|0, (8<<2)|0, (7<<2)|0, (6<<2)|0 };
    const quad = [_]u8{ makeCard(11, 0), makeCard(11, 1), makeCard(11, 2), makeCard(11, 3), makeCard(10, 0), makeCard(9, 1), makeCard(8, 2) };
    
    const royal_strength = evalHand(&royal);
    const quad_strength = evalHand(&quad);
    
    try testing.expect(royal_strength >= 9000); // Royal flush range
    try testing.expect(quad_strength >= 7000 and quad_strength < 8000); // Four of a kind range
    try testing.expect(royal_strength > quad_strength); // Royal beats quads
    std.debug.print("  ✅ Hand type ranges test passed\n", .{});
}

test "card encoding consistency" {
    std.debug.print("🔧 Testing card encoding consistency...\n", .{});
    // Test that card encoding works correctly
    const test_cards = [_]u8{ makeCard(12, 0), makeCard(0, 3) }; // Ace of spades, 2 of clubs
    
    const bits = evaluator.encodeCards(&test_cards);
    
    // Check specific bit positions
    const ace_spades_pos = 12 * 4 + 0; // Position 48
    const two_clubs_pos = 0 * 4 + 3;   // Position 3
    
    try testing.expect((bits & (@as(u64, 1) << @intCast(ace_spades_pos))) != 0); // Ace of spades
    try testing.expect((bits & (@as(u64, 1) << @intCast(two_clubs_pos))) != 0);  // 2 of clubs
    std.debug.print("  ✅ Card encoding test passed\n", .{});
}

test "evaluator produces valid results" {
    std.debug.print("🧪 Testing evaluator stability...\n", .{});
    // Test that the evaluator doesn't crash and produces reasonable results
    const hands = [_][7]u8{
        [_]u8{ makeCard(12, 0), makeCard(11, 1), makeCard(10, 2), makeCard(9, 3), makeCard(8, 0), makeCard(7, 1), makeCard(6, 2) },
        [_]u8{ makeCard(5, 0), makeCard(5, 1), makeCard(5, 2), makeCard(12, 3), makeCard(11, 0), makeCard(10, 1), makeCard(9, 2) },
        [_]u8{ makeCard(2, 0), makeCard(4, 1), makeCard(6, 2), makeCard(8, 3), makeCard(10, 0), makeCard(12, 1), makeCard(1, 2) },
    };
    
    for (hands, 0..) |hand, i| {
        const strength = evalHand(&hand);
        const hand_type = evaluator.strengthToString(strength);
        std.debug.print("  Hand {}: {} ({s})\n", .{i+1, strength, hand_type});
        
        // All evaluations should produce valid results (0-9999 range)
        try testing.expect(strength <= 9999);
        
        // Strength to string should not crash
        try testing.expect(hand_type.len > 0);
    }
    std.debug.print("  ✅ Evaluator stability test passed\n", .{});
}
