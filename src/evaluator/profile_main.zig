const std = @import("std");
const evaluator = @import("evaluator.zig");

// Re-export the core functions for assembly analysis
pub const evaluateHand = evaluator.evaluateHand;
pub const evaluateBatch4 = evaluator.evaluateBatch4;
pub const benchmarkSingle = evaluator.benchmarkSingle;
pub const benchmarkBatch = evaluator.benchmarkBatch;

// Export functions for assembly analysis
export fn bench_single_hand(hand: u64) u16 {
    return evaluateHand(hand);
}

export fn bench_batch_hands(hands: @Vector(4, u64)) @Vector(4, u16) {
    return evaluateBatch4(hands);
}

export fn bench_performance_loop() u64 {
    return benchmarkSingle(1000);
}

export fn bench_batch_performance() u64 {
    return benchmarkBatch(250); // 250 * 4 = 1000 hands
}

// Test with representative hands
pub fn main() !void {
    const test_hands = [_]u64{
        0x1F00000000000, // Royal flush in spades
        0x0000000001F00, // Royal flush in clubs
        0x123456789ABCD, // Random hand
        0x0F0F0F0F0F0F0, // Another pattern
    };

    std.debug.print("Clean Evaluator Results:\n", .{});
    for (test_hands, 0..) |hand, i| {
        const rank = evaluateHand(hand);
        std.debug.print("Hand {}: 0x{X} -> Rank: {}\n", .{ i, hand, rank });
    }

    // Batch evaluation
    const batch_hands: @Vector(4, u64) = @Vector(4, u64){ test_hands[0], test_hands[1], test_hands[2], test_hands[3] };
    const batch_results = evaluate_batch_4(batch_hands);

    std.debug.print("\nBatch evaluation results:\n", .{});
    for (0..4) |i| {
        std.debug.print("Batch[{}]: rank {}\n", .{ i, batch_results[i] });
    }

    // Performance tests
    const single_result = benchmark_single(1000);
    const batch_result = benchmark_batch(250);
    std.debug.print("\nBenchmarks:\nSingle: {}\nBatch: {}\n", .{ single_result, batch_result });
}
