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
    
    // Straight detection optimization benchmark
    print("\n=== Straight Detection Optimization Benchmark ===\n", .{});
    try benchmarkStraightDetection();
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

// Benchmark straight detection optimization
fn benchmarkStraightDetection() !void {
    // Generate test patterns covering all possible straights and non-straights
    var test_masks: [32]u16 = undefined;
    var expected_results: [32]bool = undefined;
    
    // Known straight patterns
    const straight_masks = [_]u16{
        0b1111100000000, // A-K-Q-J-T 
        0b0111110000000, // K-Q-J-T-9
        0b0011111000000, // Q-J-T-9-8
        0b0001111100000, // J-T-9-8-7
        0b0000111110000, // T-9-8-7-6
        0b0000011111000, // 9-8-7-6-5
        0b0000001111100, // 8-7-6-5-4
        0b0000000111110, // 7-6-5-4-3
        0b0000000011111, // 6-5-4-3-2
        0b1000000001111, // A-5-4-3-2 (wheel)
    };
    
    // Add straight patterns
    for (straight_masks, 0..) |mask, i| {
        test_masks[i] = mask;
        expected_results[i] = true;
    }
    
    // Add some non-straight patterns
    const non_straight_masks = [_]u16{
        0b1010101010101, // Alternating pattern
        0b1100110011001, // Pairs pattern
        0b1111000011110, // Gaps pattern
        0b0000000000001, // Single card
        0b1000000000001, // Two cards far apart
        0b1110000000000, // Three high cards
        0b0000000001110, // Three low cards
        0b1010000001010, // Scattered pattern
        0b1111000000000, // Four high, no straight
        0b0000000011110, // Four middle, no straight
        0b1000010001000, // Scattered aces pattern
        0b0101010101010, // Another alternating
    };
    
    for (non_straight_masks, 0..) |mask, i| {
        test_masks[10 + i] = mask;
        expected_results[10 + i] = false;
    }
    
    // Test correctness first
    print("Testing correctness of both implementations...\n", .{});
    var correct_lut = true;
    var correct_orig = true;
    
    for (test_masks[0..22], expected_results[0..22]) |mask, expected| {
        const result_lut = poker.checkStraight(mask);
        const result_orig = poker.checkStraightOriginal(mask);
        
        if (result_lut != expected) {
            print("  LUT ERROR: mask {b:0>13} expected {} got {}\n", .{ mask, expected, result_lut });
            correct_lut = false;
        }
        if (result_orig != expected) {
            print("  ORIG ERROR: mask {b:0>13} expected {} got {}\n", .{ mask, expected, result_orig });
            correct_orig = false;
        }
    }
    
    if (correct_lut and correct_orig) {
        print("  ✓ Both implementations correct\n", .{});
    } else {
        print("  ✗ Implementation errors found!\n", .{});
        return;
    }
    
    // Performance comparison
    const iterations = 100_000_000;
    print("Benchmarking straight detection ({} iterations)...\n", .{iterations});
    
    // Benchmark original implementation
    var dummy_result_orig: u32 = 0;
    const start_orig = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const mask = test_masks[i % test_masks.len];
        if (poker.checkStraightOriginal(mask)) {
            dummy_result_orig += 1;
        }
    }
    
    const end_orig = std.time.nanoTimestamp();
    const duration_orig = end_orig - start_orig;
    
    // Benchmark LUT implementation  
    var dummy_result_lut: u32 = 0;
    const start_lut = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const mask = test_masks[i % test_masks.len];
        if (poker.checkStraight(mask)) {
            dummy_result_lut += 1;
        }
    }
    
    const end_lut = std.time.nanoTimestamp();
    const duration_lut = end_lut - start_lut;
    
    // Results
    const ns_per_call_orig = @as(f64, @floatFromInt(duration_orig)) / @as(f64, @floatFromInt(iterations));
    const ns_per_call_lut = @as(f64, @floatFromInt(duration_lut)) / @as(f64, @floatFromInt(iterations));
    const speedup = ns_per_call_orig / ns_per_call_lut;
    
    print("Results:\n", .{});
    print("  Original (shift-mask): {d:.2}ns per call\n", .{ns_per_call_orig});
    print("  LUT implementation:    {d:.2}ns per call\n", .{ns_per_call_lut});
    print("  Speedup: {d:.2}x {s}\n", .{ speedup, if (speedup > 1.0) "(faster)" else "(slower)" });
    print("  Checksums: orig={} lut={} (prevents optimization)\n", .{ dummy_result_orig, dummy_result_lut });
}
