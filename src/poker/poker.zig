const std = @import("std");
const evaluator = @import("../evaluator/mod.zig");

// Re-export evaluator types for poker domain
pub const HandRank = evaluator.HandCategory;
pub const convertEvaluatorRank = evaluator.getHandCategory;

// Common showdown result type
pub const ShowdownResult = struct { winner: u8, tie: bool, winning_rank: HandRank };

// Basic tests
const testing = std.testing;

test "hand rank ordering" {
    try testing.expect(@intFromEnum(HandRank.high_card) < @intFromEnum(HandRank.pair));
    try testing.expect(@intFromEnum(HandRank.pair) < @intFromEnum(HandRank.straight_flush));
}

test "evaluator rank conversion" {
    // Test the conversion function
    try testing.expect(convertEvaluatorRank(1) == HandRank.straight_flush);
    try testing.expect(convertEvaluatorRank(100) == HandRank.four_of_a_kind);
    try testing.expect(convertEvaluatorRank(7000) == HandRank.high_card);
}
