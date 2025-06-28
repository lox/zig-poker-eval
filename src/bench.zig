const std = @import("std");
const evaluator = @import("evaluator/mod.zig");

pub fn main() !void {
    const print = std.debug.print;
    
    print("\x1b[1mPoker Evaluator Benchmark Suite\x1b[0m\n", .{});
    
    // Warm up and run comprehensive benchmark
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    
    // Generate test hands
    print("  Generating test hands...\n", .{});
    var test_hands: [100000]u64 = undefined;
    for (&test_hands) |*hand| {
        hand.* = evaluator.generateRandomHand(&rng);
    }
    
    // Warm up caches
    print("  Warming up caches...\n", .{});
    for (0..1000) |i| {
        const idx = i * 4;
        if (idx + 3 < test_hands.len) {
            const batch = @Vector(4, u64){ test_hands[idx], test_hands[idx+1], test_hands[idx+2], test_hands[idx+3] };
            _ = evaluator.evaluateBatch4(batch);
        }
    }
    
    // Run performance benchmark
    print("  Running benchmark...\n", .{});
    
    const start = std.time.nanoTimestamp();
    var sum: u64 = 0;
    
    // Batch processing
    var i: usize = 0;
    while (i + 4 <= test_hands.len) : (i += 4) {
        const batch = @Vector(4, u64){ test_hands[i], test_hands[i+1], test_hands[i+2], test_hands[i+3] };
        const results = evaluator.evaluateBatch4(batch);
        sum +%= results[0] + results[1] + results[2] + results[3];
    }
    
    const end = std.time.nanoTimestamp();
    const total_time = @as(f64, @floatFromInt(end - start));
    const hands_evaluated = @as(f64, @floatFromInt(test_hands.len));
    const ns_per_hand = total_time / hands_evaluated;
    
    print("\n\x1b[1mBenchmark Results\x1b[0m\n", .{});
    print("  Total hands:      {}\n", .{test_hands.len});
    print("  Time per hand:    {d:.2} ns\n", .{ns_per_hand});
    print("  Hands per second: {d:.0}\n", .{1e9 / ns_per_hand});
    
    // Run accuracy validation
    print("\n\x1b[1mAccuracy Validation\x1b[0m\n", .{});
    var validation_matches: u32 = 0;
    const validation_hands = @min(1000, test_hands.len);
    
    for (0..validation_hands) |idx| {
        const fast_result = evaluator.evaluateHand(test_hands[idx]);
        const slow_result = evaluator.slow.evaluateHand(test_hands[idx]);
        
        if (fast_result == slow_result) {
            validation_matches += 1;
        }
    }
    
    const accuracy = @as(f64, @floatFromInt(validation_matches)) / @as(f64, @floatFromInt(validation_hands)) * 100.0;
    print("  Accuracy:         {d:.1}% ({}/{})\n", .{ accuracy, validation_matches, validation_hands });
    
    if (accuracy < 100.0) {
        print("  \x1b[31mWARNING: Accuracy below 100%!\x1b[0m\n", .{});
        std.process.exit(1);
    } else {
        print("  \x1b[32m✓ Perfect accuracy\x1b[0m\n", .{});
    }
    
    std.mem.doNotOptimizeAway(sum);
}