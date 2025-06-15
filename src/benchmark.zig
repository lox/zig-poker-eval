const std = @import("std");
const poker = @import("poker.zig");

// Generate random 7-card hands for fair benchmarking
pub fn generateRandomHands(allocator: std.mem.Allocator, count: u32, seed: u64) ![]poker.Hand {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    
    const hands = try allocator.alloc(poker.Hand, count);
    
    for (hands) |*hand| {
        hand.* = generateRandomHand(random);
    }
    
    return hands;
}

fn generateRandomHand(random: std.Random) poker.Hand {
    var hand = poker.Hand.init();
    var used_cards = std.StaticBitSet(52).initEmpty();
    
    // Generate 7 unique random cards
    var cards_added: u8 = 0;
    while (cards_added < 7) {
        const card_idx = random.uintLessThan(u8, 52);
        if (!used_cards.isSet(card_idx)) {
            used_cards.set(card_idx);
            
            // Convert card index to rank/suit
            const rank: u8 = (card_idx / 4) + 2; // 0-51 -> ranks 2-14
            const suit: u2 = @intCast(card_idx % 4);
            
            hand.addCard(poker.Card.init(rank, suit));
            cards_added += 1;
        }
    }
    
    return hand;
}

// Generate torture case hands (edge cases) using clean card notation
pub fn generateTortureCases(allocator: std.mem.Allocator) ![]poker.Hand {
    const hands = try allocator.alloc(poker.Hand, 11);
    
    // Using runtime parseCards for clean, readable card definitions  
    hands[0] = poker.parseCards("AsKsQsJsTs2h3d") catch unreachable;  // Royal flush + 2 random
    hands[1] = poker.parseCards("9h8h7h6h5h2s3d") catch unreachable;  // Straight flush + 2 random  
    hands[2] = poker.parseCards("AhAsAdAc2h3s4d") catch unreachable;  // Four of a kind + 3 random
    hands[3] = poker.parseCards("AhAsAdKhKs2c3d") catch unreachable;  // Full house + 2 random
    hands[4] = poker.parseCards("AhQhTh8h6h2s3d") catch unreachable;  // Flush + 2 random
    hands[5] = poker.parseCards("AhKsQdJcTh2s3d") catch unreachable;  // Straight + 2 random
    hands[6] = poker.parseCards("AhAsAdKhQs2c3d") catch unreachable;  // Three of a kind + 4 random
    hands[7] = poker.parseCards("AhAsKhKsQh2c3d") catch unreachable;  // Two pair + 3 random
    hands[8] = poker.parseCards("AhAsKhQsJd2c3d") catch unreachable;  // One pair + 5 random
    hands[9] = poker.parseCards("AhKsQdJc9h7s2d") catch unreachable;  // High card + 6 random
    hands[10] = poker.parseCards("Ah5s4d3c2h6s7d") catch unreachable; // Wheel straight (A-2-3-4-5) + 2 random
    
    return hands;
}

// Comprehensive benchmarking suite
pub const BenchmarkResults = struct {
    realistic_hands_per_sec: f64,
    random_hands_per_sec: f64, // Kept for compatibility
    straight_detection_ns: f64, // Kept for compatibility
    memory_usage_mb: f64,
};

// Run focused performance benchmark
pub fn runComprehensiveBenchmark(allocator: std.mem.Allocator) !BenchmarkResults {
    const print = std.debug.print;
    
    print("=== Performance Benchmark ===\n", .{});
    
    // Realistic performance test (primary metric)
    print("\n1. Realistic Performance Test\n", .{});
    print("   (10M unique hands with memory pressure - real-world scenario)\n", .{});
    const realistic_perf = try benchmarkRealistic(allocator);
    
    // Memory usage calculation
    const memory_mb = 10_000_000.0 * @as(f64, @sizeOf(poker.Hand)) / 1024.0 / 1024.0;
    
    print("\n=== Performance Summary ===\n", .{});
    print("  Primary Metric: {d:.2}M hands/sec\n", .{realistic_perf / 1_000_000});
    print("  Memory (10M hands): {d:.1}MB\n", .{memory_mb});
    
    return BenchmarkResults{
        .realistic_hands_per_sec = realistic_perf,
        .random_hands_per_sec = 0.0, // Removed secondary metric
        .straight_detection_ns = 0.0, // Removed micro-benchmark
        .memory_usage_mb = memory_mb,
    };
}

// Realistic performance benchmark (diverse hands with memory pressure)
fn benchmarkRealistic(allocator: std.mem.Allocator) !f64 {
    const print = std.debug.print;
    const trials = 10_000_000;
    
    print("   Generating {} unique random hands...\n", .{trials});
    const hands = try generateRandomHands(allocator, trials, 42);
    defer allocator.free(hands);
    
    print("   Benchmarking evaluation...\n", .{});
    const start = std.time.nanoTimestamp();
    var dummy_result: poker.HandRank = .high_card;
    
    for (hands) |hand| {
        const result = hand.evaluate();
        if (@intFromEnum(result) > @intFromEnum(dummy_result)) {
            dummy_result = result;
        }
    }
    
    const end = std.time.nanoTimestamp();
    const duration_s = @as(f64, @floatFromInt(end - start)) / 1_000_000_000.0;
    const hands_per_sec = @as(f64, @floatFromInt(trials)) / duration_s;
    
    print("   Result: {d:.2}M hands/sec ({d:.0}ns per hand)\n", .{ hands_per_sec / 1_000_000, @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(trials)) });
    return hands_per_sec;
}

// Random hands performance benchmark (realistic scenario)
fn benchmarkRandomHands(allocator: std.mem.Allocator) !f64 {
    const print = std.debug.print;
    const trials = 1_000_000;
    
    print("   Generating {} diverse random hands...\n", .{trials});
    const hands = try generateRandomHands(allocator, trials, 456);
    defer allocator.free(hands);
    
    print("   Benchmarking evaluation...\n", .{});
    const start = std.time.nanoTimestamp();
    var dummy_result: poker.HandRank = .high_card;
    
    for (hands) |hand| {
        const result = hand.evaluate();
        if (@intFromEnum(result) > @intFromEnum(dummy_result)) {
            dummy_result = result;
        }
    }
    
    const end = std.time.nanoTimestamp();
    const duration_s = @as(f64, @floatFromInt(end - start)) / 1_000_000_000.0;
    const hands_per_sec = @as(f64, @floatFromInt(trials)) / duration_s;
    
    print("   Result: {d:.2}M hands/sec ({d:.0}ns per hand)\n", .{ hands_per_sec / 1_000_000, @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(trials)) });
    return hands_per_sec;
}

// Micro-benchmark for straight detection
fn benchmarkStraightDetection() !f64 {
    const print = std.debug.print;
    
    // Generate test patterns
    var test_masks: [32]u16 = undefined;
    const straight_masks = [_]u16{
        0b1111100000000, 0b0111110000000, 0b0011111000000, 0b0001111100000, 0b0000111110000,
        0b0000011111000, 0b0000001111100, 0b0000000111110, 0b0000000011111, 0b1000000001111,
    };
    const non_straight_masks = [_]u16{
        0b1010101010101, 0b1100110011001, 0b1111000011110, 0b0000000000001, 0b1000000000001,
        0b1110000000000, 0b0000000001110, 0b1010000001010, 0b1111000000000, 0b0000000011110,
    };
    
    for (straight_masks, 0..) |mask, i| test_masks[i] = mask;
    for (non_straight_masks, 0..) |mask, i| test_masks[10 + i] = mask;
    
    // Benchmark straight detection
    const iterations = 100_000_000;
    print("   Testing straight detection ({} iterations)...\n", .{iterations});
    
    var dummy_result: u32 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const mask = test_masks[i % test_masks.len];
        if (poker.checkStraight(mask)) {
            dummy_result += 1;
        }
    }
    
    const end = std.time.nanoTimestamp();
    const ns_per_call = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iterations));
    
    print("   Result: {d:.2}ns per call\n", .{ns_per_call});
    print("   Checksum: {} (prevents optimization)\n", .{dummy_result});
    return ns_per_call;
}
