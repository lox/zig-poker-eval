const std = @import("std");
const evaluator = @import("evaluator.zig");

pub fn main() !void {
    const print = std.debug.print;

    print("Zig Poker Hand Evaluator\n", .{});

    // Test a few hands
    const royal_flush = evaluator.makeCard(3, 12) | evaluator.makeCard(3, 11) |
        evaluator.makeCard(3, 10) | evaluator.makeCard(3, 9) |
        evaluator.makeCard(3, 8) | evaluator.makeCard(0, 0) |
        evaluator.makeCard(1, 1);

    print("Royal flush rank: {}\n", .{evaluator.evaluateHand(royal_flush)});

    const four_aces = evaluator.makeCard(0, 12) | evaluator.makeCard(1, 12) |
        evaluator.makeCard(2, 12) | evaluator.makeCard(3, 12) |
        evaluator.makeCard(0, 11) | evaluator.makeCard(1, 10) |
        evaluator.makeCard(2, 9);

    print("Four aces rank: {}\n", .{evaluator.evaluateHand(four_aces)});
}

// Import all tests from all modules
test {
    _ = @import("evaluator.zig");
    _ = @import("chd.zig");
    _ = @import("simd_evaluator.zig");
    _ = @import("correctness.zig");
}
