const std = @import("std");
const evaluator = @import("evaluator");

pub fn main() !void {
    const print = std.debug.print;

    print("Zig Poker Hand Evaluator\n", .{});

    // Test hands using the slow evaluator's makeCard function
    const royal_flush = evaluator.slow.makeCard(3, 12) | evaluator.slow.makeCard(3, 11) |
        evaluator.slow.makeCard(3, 10) | evaluator.slow.makeCard(3, 9) |
        evaluator.slow.makeCard(3, 8) | evaluator.slow.makeCard(0, 0) |
        evaluator.slow.makeCard(1, 1);

    const slow_royal = evaluator.slow.evaluateHand(royal_flush);
    const fast_royal = evaluator.evaluateHand(royal_flush);
    print("Royal flush - Slow: {}, Fast: {}, Match: {}\n", .{ slow_royal, fast_royal, slow_royal == fast_royal });

    const four_aces = evaluator.slow.makeCard(0, 12) | evaluator.slow.makeCard(1, 12) |
        evaluator.slow.makeCard(2, 12) | evaluator.slow.makeCard(3, 12) |
        evaluator.slow.makeCard(0, 11) | evaluator.slow.makeCard(1, 10) |
        evaluator.slow.makeCard(2, 9);

    const slow_aces = evaluator.slow.evaluateHand(four_aces);
    const fast_aces = evaluator.evaluateHand(four_aces);
    print("Four aces - Slow: {}, Fast: {}, Match: {}\n", .{ slow_aces, fast_aces, slow_aces == fast_aces });
}

// Import tests from evaluator module
test {
    _ = evaluator;
}
