const std = @import("std");
const evaluator = @import("evaluator.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    try runEvaluatorBenchmark(allocator);
}

pub fn runEvaluatorBenchmark(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;

    print("✋ Ultra-Fast Poker Hand Evaluator Benchmark\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

    // Generate fixed set of random hands
    const hand_count = 10000;
    print("Generating {} random hands...\n", .{hand_count});
    
    const hands = try allocator.alloc(u64, hand_count);
    defer allocator.free(hands);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    
    for (hands) |*hand| {
        // Generate 7 unique random cards (0-51)
        var cards: [7]u8 = undefined;
        var used = std.bit_set.IntegerBitSet(52).initEmpty();
        
        for (0..7) |i| {
            var card: u8 = undefined;
            while (true) {
                card = @intCast(random.uintLessThan(u32, 52));
                if (!used.isSet(card)) {
                    used.set(card);
                    break;
                }
            }
            cards[i] = card;
        }
        
        hand.* = evaluator.encodeCards(&cards);
    }

    // Warm-up run
    var dummy: u64 = 0;
    for (hands) |hand| {
        const result = evaluator.eval7(hand);
        dummy +%= result;
    }

    // Main benchmark - multiple runs
    const runs = 3;
    var total_ops: u64 = 0;
    var total_ns: u64 = 0;
    var run_results = try allocator.alloc(f64, runs);
    defer allocator.free(run_results);

    for (0..runs) |run| {
        const start = std.time.nanoTimestamp();
        var ops: u64 = 0;
        const target_duration_ns = 1_000_000_000; // 1 second target

        // Run for approximately 1 second
        while (std.time.nanoTimestamp() - start < target_duration_ns) {
            for (hands) |hand| {
                const result = evaluator.eval7(hand);
                dummy +%= result;
                ops += 1;
            }
        }

        const end = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end - start));
        const ns_per_op = @as(f64, @floatFromInt(duration_ns)) / @as(f64, @floatFromInt(ops));

        print("Run {}: {} ops, {d:.2} ns/op\n", .{ run + 1, ops, ns_per_op });
        
        run_results[run] = ns_per_op;
        total_ops += ops;
        total_ns += duration_ns;
    }

    // Calculate statistics
    const avg_ns_per_op = @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(total_ops));
    const evaluations_per_sec = 1_000_000_000.0 / avg_ns_per_op;

    // Calculate standard deviation
    var variance: f64 = 0;
    for (run_results) |result| {
        const diff = result - avg_ns_per_op;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(runs));
    const std_dev = @sqrt(variance);
    const coefficient_of_variation = (std_dev / avg_ns_per_op) * 100.0;

    // Find min/max
    var min_result = run_results[0];
    var max_result = run_results[0];
    for (run_results[1..]) |result| {
        if (result < min_result) min_result = result;
        if (result > max_result) max_result = result;
    }

    print("\n📊 Performance Summary\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    print("Average:     {d:.2} ns/op (across {} runs)\n", .{ avg_ns_per_op, runs });
    print("Std Dev:     {d:.2} ns/op ({d:.1}% CV)\n", .{ std_dev, coefficient_of_variation });
    print("Min/Max:     {d:.2} - {d:.2} ns/op\n", .{ min_result, max_result });
    print("Throughput:  {d:.1}M evaluations/second\n", .{ evaluations_per_sec / 1_000_000.0 });

    print("\n🎯 Performance Comparison\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    print("OMPEval target:  1.3 ns/op\n", .{});
    print("Ultra-fast:      {d:.2} ns/op\n", .{avg_ns_per_op});
    print("Factor slower:   {d:.1}x\n", .{avg_ns_per_op / 1.3});

    if (avg_ns_per_op <= 1.3) {
        print("Status: ✅ TARGET ACHIEVED!\n", .{});
    } else if (avg_ns_per_op <= 5.0) {
        print("Status: 🟡 Close to target (within 4x)\n", .{});
    } else {
        print("Status: 🔴 Needs more optimization\n", .{});
    }

    // Variability assessment
    if (coefficient_of_variation < 5.0) {
        print("Stability: ✅ Very stable (CV < 5%)\n", .{});
    } else if (coefficient_of_variation < 10.0) {
        print("Stability: 🟡 Moderately stable (CV < 10%)\n", .{});
    } else {
        print("Stability: 🔴 High variability (CV > 10%)\n", .{});
    }

    print("Checksum (prevent optimization): {}\n", .{dummy});
}
