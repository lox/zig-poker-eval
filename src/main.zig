const std = @import("std");
const evaluator = @import("evaluator.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    
    // Test hands using correct encoding: [rank:4][suit:2] where rank 0=2, 1=3, ..., 12=A
    // and suit 0=♠, 1=♥, 2=♦, 3=♣
    const test_hands = [_][7]u8{
        // Royal flush in spades: As(12,0) Ks(11,0) Qs(10,0) Js(9,0) 10s(8,0) 9s(7,0) 8s(6,0)  
        .{ (12<<2)|0, (11<<2)|0, (10<<2)|0, (9<<2)|0, (8<<2)|0, (7<<2)|0, (6<<2)|0 },
        // Four aces: Ah Ac Ad As + Kh Qh Jh
        .{ (12<<2)|1, (12<<2)|3, (12<<2)|2, (12<<2)|0, (11<<2)|1, (10<<2)|1, (9<<2)|1 },
        // Full house: 8s 8h 8d + 3c 3s + Ah Kh
        .{ (6<<2)|0, (6<<2)|1, (6<<2)|2, (1<<2)|3, (1<<2)|0, (12<<2)|1, (11<<2)|1 },
        // Flush in hearts: Ah Kh Qh Jh 9h 7h 5h
        .{ (12<<2)|1, (11<<2)|1, (10<<2)|1, (9<<2)|1, (7<<2)|1, (5<<2)|1, (3<<2)|1 },
        // Two pair: As Ah + Ks Kh + Qd Jc 9s
        .{ (12<<2)|0, (12<<2)|1, (11<<2)|0, (11<<2)|1, (10<<2)|2, (9<<2)|3, (7<<2)|0 },
    };
    
    std.debug.print("7-Card Poker Hand Evaluator\n", .{});
    std.debug.print("==========================\n\n", .{});
    
    for (test_hands, 0..) |hand, i| {
        const bits = evaluator.encodeCards(&hand);
        const strength = evaluator.eval7(bits);
        const hand_type = evaluator.strengthToString(strength);
        
        std.debug.print("Hand {}: {s} (strength: {})\n", .{ i + 1, hand_type, strength });
    }
    
    // Quick correctness test
    const royal_bits = evaluator.encodeCards(&test_hands[0]);
    const royal_strength = evaluator.eval7(royal_bits);
    
    if (royal_strength >= 9000) { // Royal flush strength
        std.debug.print("\n✓ Royal flush correctly identified\n", .{});
    } else {
        std.debug.print("\n✗ Royal flush evaluation failed\n", .{});
    }
}

test "basic evaluation" {
    // Test that high cards are ranked correctly
    const high_card = evaluator.encodeCards(&[_]u8{ 0x33, 0x2f, 0x2b, 0x27, 0x1f, 0x17, 0x0f });
    const strength = evaluator.eval7(high_card);
    
    // Should be a flush (all hearts)
    try std.testing.expect(strength >= 5000); // Flush strength
}

test "pair evaluation" {
    const pair = evaluator.encodeCards(&[_]u8{ 0x33, 0x30, 0x2f, 0x2b, 0x27, 0x1f, 0x17 });
    const strength = evaluator.eval7(pair);
    
    // Should be at least a pair
    try std.testing.expect(strength >= 1000); // Pair strength
}
