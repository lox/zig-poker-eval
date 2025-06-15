const std = @import("std");
const poker = @import("poker.zig");
pub fn runEvaluatorBenchmark(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("=== Benchmark  ===\n", .{});

    // Generate fixed set of hands like Go benchmark
    const hand_count = 10000;
    print("Generating {} random hands...\n", .{hand_count});
    const hands = try poker.generateRandomHands(allocator, hand_count, 42);
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
