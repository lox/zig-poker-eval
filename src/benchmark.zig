const std = @import("std");
const poker = @import("poker.zig");

// Generate random 7-card hands matching Go's methodology
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

pub fn runEvaluatorBenchmark(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("=== Benchmark  ===\n", .{});

    // Generate fixed set of hands like Go benchmark
    const hand_count = 10000;
    print("Generating {} random hands...\n", .{hand_count});
    const hands = try generateRandomHands(allocator, hand_count, 42);
    defer allocator.free(hands);

    // Warm-up run
    var dummy: poker.HandRank = .high_card;
    for (hands) |hand| {
        const result = hand.evaluate();
        if (@intFromEnum(result) > @intFromEnum(dummy)) {
            dummy = result;
        }
    }

    // Main benchmark - multiple runs like Go's -count=3
    const runs = 3;
    var total_ops: u64 = 0;
    var total_ns: u64 = 0;

    for (0..runs) |run| {
        const start = std.time.nanoTimestamp();
        var ops: u64 = 0;
        const target_duration_ns = 1_000_000_000; // 1 second target

        // Run for approximately 1 second
        while (std.time.nanoTimestamp() - start < target_duration_ns) {
            for (hands) |hand| {
                const result = hand.evaluate();
                if (@intFromEnum(result) > @intFromEnum(dummy)) {
                    dummy = result;
                }
                ops += 1;
            }
        }

        const end = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end - start));
        const ns_per_op = duration_ns / ops;

        print("Run {}: {} ops, {d:.2} ns/op\n", .{ run + 1, ops, @as(f64, @floatFromInt(ns_per_op)) });

        total_ops += ops;
        total_ns += duration_ns;
    }

    const avg_ns_per_op = total_ns / total_ops;
    const avg_ns_per_op_f64 = @as(f64, @floatFromInt(avg_ns_per_op));
    const evaluations_per_sec = 1_000_000_000.0 / avg_ns_per_op_f64;

    print("\n=== Performance Summary ===\n", .{});
    print("{d:.2} ns/op (average across {} runs)\n", .{ avg_ns_per_op_f64, runs });
    print("{d:.1}M evaluations/second\n", .{evaluations_per_sec / 1_000_000.0});
}
