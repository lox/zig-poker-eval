const std = @import("std");
const print = std.debug.print;
const poker = @import("poker.zig");
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});

    // Quick functionality test
    print("\n=== Quick Test ===\n", .{});
    
    // Clean card notation like Go's deck.MustParseCards()
    const royal_flush = try poker.parseCards("AsKsQsJsTs2h3d"); // Royal flush + 2 random
    const pair_hand = try poker.parseCards("AhAs2h4s6d8cTh");  // Pair of aces + 5 random

    print("Royal flush (AsKsQsJsTs2h3d): {}\n", .{royal_flush.evaluate()});
    print("Pair of aces (AhAs2h4s6d8cTh): {}\n", .{pair_hand.evaluate()});

    // Performance benchmark
    print("\n=== Performance Benchmark ===\n", .{});
    try benchmarkEvaluator(allocator);
    
    // Torture case benchmark 
    print("\n=== Torture Case Benchmark ===\n", .{});
    try benchmarkTortureCases(allocator);
}

fn benchmarkEvaluator(allocator: std.mem.Allocator) !void {
    const trials = 10_000_000;
    print("Generating {} random hands...\n", .{trials});
    
    const hands = try benchmark.generateRandomHands(allocator, trials, 42);
    defer allocator.free(hands);
    
    print("Benchmarking evaluation ({} trials)...\n", .{trials});
    
    const start = std.time.nanoTimestamp();
    var dummy_result: poker.HandRank = .high_card;
    
    for (hands) |hand| {
        const result = hand.evaluate();
        // Prevent optimization by doing something with the result
        if (@intFromEnum(result) > @intFromEnum(dummy_result)) {
            dummy_result = result;
        }
    }
    
    const end = std.time.nanoTimestamp();
    const duration = end - start;
    const duration_s = @as(f64, @floatFromInt(duration)) / 1_000_000_000.0;
    const hands_per_second = @as(f64, @floatFromInt(trials)) / duration_s;
    
    print("Results:\n", .{});
    print("  Time: {d:.3}s\n", .{duration_s});
    print("  Rate: {d:.0} hands/second\n", .{hands_per_second});
    print("  Avg: {d:.0}ns per hand\n", .{@as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(trials))});
    print("  Dummy result: {} (prevents optimization)\n", .{dummy_result});
}

fn benchmarkTortureCases(allocator: std.mem.Allocator) !void {
    const torture_hands = try benchmark.generateTortureCases(allocator);
    defer allocator.free(torture_hands);
    
    print("Testing {} edge case hands cycling through them...\n", .{torture_hands.len});
    
    const trials = 10_000_000;
    const start = std.time.nanoTimestamp();
    
    // Cycle through all torture cases to prevent optimization
    var result_sum: u32 = 0;
    for (0..trials) |i| {
        const hand = torture_hands[i % torture_hands.len];
        const result = hand.evaluate();
        result_sum += @intFromEnum(result);
    }
    
    const end = std.time.nanoTimestamp();
    const duration = end - start;
    const duration_s = @as(f64, @floatFromInt(duration)) / 1_000_000_000.0;
    const hands_per_second = @as(f64, @floatFromInt(trials)) / duration_s;
    const avg_ns = @as(f64, @floatFromInt(duration)) / @as(f64, @floatFromInt(trials));
    
    print("  Mixed torture cases: {d:.0} hands/sec ({d:.0}ns)\n", .{ hands_per_second, avg_ns });
    print("  Checksum: {} (prevents optimization)\n", .{result_sum});
    
    // Also show what each case evaluates to
    print("  Hand type verification:\n", .{});
    const hand_names = [_][]const u8{
        "Royal flush", "Straight flush", "Four of a kind", "Full house",
        "Flush", "Straight", "Three of a kind", "Two pair", 
        "One pair", "High card", "Wheel straight"
    };
    
    for (torture_hands, 0..) |hand, i| {
        print("    {s} -> {}\n", .{ hand_names[i], hand.evaluate() });
    }
}

// Tests
test "card creation and properties" {
    const card = poker.createCard(.spades, .ace);
    try std.testing.expect(card.getRank() == 14);
    try std.testing.expect(card.getSuit() == 1);
}

test "hand rank ordering" {
    try std.testing.expect(@intFromEnum(poker.HandRank.high_card) < @intFromEnum(poker.HandRank.pair));
    try std.testing.expect(@intFromEnum(poker.HandRank.pair) < @intFromEnum(poker.HandRank.straight_flush));
}

test "known hand evaluations" {
    // Royal flush
    const royal_flush = poker.createHand(&.{
        .{ .spades, .ace },
        .{ .spades, .king },
        .{ .spades, .queen },
        .{ .spades, .jack },
        .{ .spades, .ten },
        .{ .hearts, .two },
        .{ .hearts, .three },
    });
    try std.testing.expect(royal_flush.evaluate() == .straight_flush);

    // Pair of aces
    const pair_hand = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .ace },
        .{ .hearts, .two },
        .{ .spades, .four },
        .{ .diamonds, .six },
        .{ .clubs, .eight },
        .{ .hearts, .ten },
    });
    try std.testing.expect(pair_hand.evaluate() == .pair);

    // High card
    const high_card = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .king },
        .{ .hearts, .queen },
        .{ .spades, .ten },
        .{ .diamonds, .eight },
        .{ .clubs, .six },
        .{ .hearts, .four },
    });
    try std.testing.expect(high_card.evaluate() == .high_card);
}

test "benchmark random hands" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const hands = try benchmark.generateRandomHands(allocator, 1000, 123);
    defer allocator.free(hands);
    
    for (hands) |hand| {
        const result = hand.evaluate();
        try std.testing.expect(@intFromEnum(result) >= 1);
        try std.testing.expect(@intFromEnum(result) <= 9);
    }
}
