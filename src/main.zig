const std = @import("std");
const evaluator = @import("evaluator.zig");
const slow_evaluator = @import("slow_evaluator.zig");

pub fn main() !void {
    const print = std.debug.print;

    print("Zig Poker Hand Evaluator\n", .{});

    // Test hands using the slow evaluator's makeCard function
    const royal_flush = slow_evaluator.makeCard(3, 12) | slow_evaluator.makeCard(3, 11) |
        slow_evaluator.makeCard(3, 10) | slow_evaluator.makeCard(3, 9) |
        slow_evaluator.makeCard(3, 8) | slow_evaluator.makeCard(0, 0) |
        slow_evaluator.makeCard(1, 1);

    const slow_royal = slow_evaluator.evaluateHand(royal_flush);
    const fast_royal = evaluator.evaluateHand(royal_flush);
    print("Royal flush - Slow: {}, Fast: {}, Match: {}\n", .{ slow_royal, fast_royal, slow_royal == fast_royal });

    const four_aces = slow_evaluator.makeCard(0, 12) | slow_evaluator.makeCard(1, 12) |
        slow_evaluator.makeCard(2, 12) | slow_evaluator.makeCard(3, 12) |
        slow_evaluator.makeCard(0, 11) | slow_evaluator.makeCard(1, 10) |
        slow_evaluator.makeCard(2, 9);

    const slow_aces = slow_evaluator.evaluateHand(four_aces);
    const fast_aces = evaluator.evaluateHand(four_aces);
    print("Four aces - Slow: {}, Fast: {}, Match: {}\n", .{ slow_aces, fast_aces, slow_aces == fast_aces });
}

// Import tests from all modules
test {
    _ = @import("evaluator.zig");
    _ = @import("validation.zig");
}
