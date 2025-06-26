const std = @import("std");
const simd_evaluator = @import("simd_evaluator.zig");
const slow_evaluator = @import("slow_evaluator");
const validation = @import("validation.zig");

// Helper functions for rigorous benchmarking
const BATCH_SIZE = 16; // Fixed batch size for current implementation

fn warmupCaches(test_hands: []const u64, simd_eval: *const simd_evaluator.SIMDEvaluator) void {
    // Touch lookup tables (if available)
    // Note: @import will fail at compile time if tables.zig doesn't exist
    // const tables = @import("tables.zig");
    // if (@hasDecl(tables, "rank_patterns")) {
    //     for (tables.rank_patterns[0..@min(16384, tables.rank_patterns.len)]) |pattern| {
    //         std.mem.doNotOptimizeAway(pattern);
    //     }
    // }

    // Touch first portion of hands by evaluating them in batches
    const warmup_hands = @min(65536, test_hands.len); // 64K hands max
    var i: usize = 0;
    while (i + BATCH_SIZE <= warmup_hands) {
        const batch = simd_evaluator.VecU64{
            test_hands[i], test_hands[i+1], test_hands[i+2], test_hands[i+3],
            test_hands[i+4], test_hands[i+5], test_hands[i+6], test_hands[i+7],
            test_hands[i+8], test_hands[i+9], test_hands[i+10], test_hands[i+11],
            test_hands[i+12], test_hands[i+13], test_hands[i+14], test_hands[i+15]
        };
        _ = simd_eval.evaluate_batch(batch);
        i += BATCH_SIZE;
    }
}

fn clearCaches(use_purge: bool) void {
    if (!use_purge) return;

    var child = std.process.Child.init(&[_][]const u8{ "sudo", "purge" }, std.heap.page_allocator);
    _ = child.spawnAndWait() catch return;
}

fn calculateCV(times: []const f64) f64 {
    var sum: f64 = 0;
    for (times) |time| {
        sum += time;
    }
    const mean = sum / @as(f64, @floatFromInt(times.len));

    var variance: f64 = 0;
    for (times) |time| {
        const diff = time - mean;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(times.len));

    const std_dev = @sqrt(variance);
    return std_dev / mean;
}

fn runSingleBenchmark(iterations: u32, test_hands: []const u64, simd_eval: *const simd_evaluator.SIMDEvaluator) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    const total_hands = iterations * BATCH_SIZE;

    var hand_idx: usize = 0;
    for (0..iterations) |_| {
        // Create batch from consecutive hands
        const batch = simd_evaluator.VecU64{
            test_hands[hand_idx % test_hands.len], test_hands[(hand_idx+1) % test_hands.len],
            test_hands[(hand_idx+2) % test_hands.len], test_hands[(hand_idx+3) % test_hands.len],
            test_hands[(hand_idx+4) % test_hands.len], test_hands[(hand_idx+5) % test_hands.len],
            test_hands[(hand_idx+6) % test_hands.len], test_hands[(hand_idx+7) % test_hands.len],
            test_hands[(hand_idx+8) % test_hands.len], test_hands[(hand_idx+9) % test_hands.len],
            test_hands[(hand_idx+10) % test_hands.len], test_hands[(hand_idx+11) % test_hands.len],
            test_hands[(hand_idx+12) % test_hands.len], test_hands[(hand_idx+13) % test_hands.len],
            test_hands[(hand_idx+14) % test_hands.len], test_hands[(hand_idx+15) % test_hands.len]
        };
        
        const results = simd_eval.evaluate_batch(batch);
        for (0..BATCH_SIZE) |j| {
            checksum +%= results[j];
        }
        hand_idx += BATCH_SIZE;
    }

    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);

    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(total_hands));
}

fn benchmarkDummyEvaluator(iterations: u32, test_hands: []const u64) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    const total_hands = iterations * BATCH_SIZE;

    var hand_idx: usize = 0;
    for (0..iterations) |_| {
        for (0..BATCH_SIZE) |j| {
            checksum +%= @popCount(test_hands[(hand_idx + j) % test_hands.len]); // Trivial operation
        }
        hand_idx += BATCH_SIZE;
    }

    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);

    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(total_hands));
}

fn validateCorrectness(test_hands: []const u64) !bool {
    const simd_eval = simd_evaluator.SIMDEvaluator.init();
    var matches: u32 = 0;
    var total: u32 = 0;

    // Validate first 16K hands in batches
    const validation_hands = @min(16000, test_hands.len);
    var i: usize = 0;
    while (i + BATCH_SIZE <= validation_hands) {
        const batch = simd_evaluator.VecU64{
            test_hands[i], test_hands[i+1], test_hands[i+2], test_hands[i+3],
            test_hands[i+4], test_hands[i+5], test_hands[i+6], test_hands[i+7],
            test_hands[i+8], test_hands[i+9], test_hands[i+10], test_hands[i+11],
            test_hands[i+12], test_hands[i+13], test_hands[i+14], test_hands[i+15]
        };
        
        const fast_results = simd_eval.evaluate_batch(batch);

        for (0..BATCH_SIZE) |j| {
            const slow_result = slow_evaluator.evaluateHand(test_hands[i + j]);
            const fast_result = fast_results[j];

            if (slow_result == fast_result) {
                matches += 1;
            }
            total += 1;
        }
        i += BATCH_SIZE;
    }

    const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total));
    std.debug.print("\x1b[1mCorrectness Validation\x1b[0m\n", .{});
    std.debug.print("  Matches:   {}/{}\n", .{ matches, total });
    std.debug.print("  Accuracy:  {d:.2}%\n\n", .{accuracy * 100.0});

    // Require 100% accuracy (updated requirement)
    if (accuracy < 1.0) {
        return error.AccuracyTooLow;
    }
    
    return true;
}

// Test the SIMD evaluator
pub fn testSimdEvaluator() !void {
    _ = std.mem.Allocator;
    const print = std.debug.print;

    print("\x1b[1mSIMD Evaluator Test\x1b[0m\n", .{});

    const simd_eval = simd_evaluator.SIMDEvaluator.init();

    // Generate test batch
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = validation.generateRandomHandBatch(&rng);

    // Evaluate batch
    const batch_results = simd_eval.evaluate_batch(batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    for (0..BATCH_SIZE) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = slow_evaluator.evaluateHand(hand);

        if (batch_result == single_result) {
            matches += 1;
        }

        if (i < 5) { // Show first 5 for debugging
            print("Hand {d}: batch={d}, single={d}, match={s}\n", .{ i, batch_result, single_result, if (batch_result == single_result) "✓" else "✗" });
        }
    }

    print("\nAccuracy: {}/{} ({d:.1}%)\n", .{ matches, BATCH_SIZE, @as(f64, @floatFromInt(matches)) * 100.0 / @as(f64, @floatFromInt(BATCH_SIZE)) });
    print("{s} SIMD evaluator test complete\n\n", .{if (matches == BATCH_SIZE) "✓" else "✗"});
}

// Benchmark the SIMD evaluator with rigorous methodology
pub fn benchmarkSimdEvaluator(iterations: u32) !void {
    const print = std.debug.print;

    print("\x1b[1mSIMD Evaluator Benchmark\x1b[0m\n", .{});

    const simd_eval = simd_evaluator.SIMDEvaluator.init();

    // Generate test hands (instead of batches)
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const num_test_hands = 1_600_000; // 100K batches worth of hands
    var test_hands: [num_test_hands]u64 = undefined;
    for (&test_hands) |*hand| {
        hand.* = validation.generateRandomHand(&rng);
    }

    // Validate correctness
    print("\n  Validating correctness...\n", .{});
    const is_correct = validateCorrectness(&test_hands) catch |err| {
        print("  \u{274C} ERROR: Correctness validation failed: {}\n\n", .{err});
        return;
    };
    if (is_correct) {
        print("  ✓ Correctness validation passed\n", .{});
    }

    // Cache warmup
    print("  Warming up caches...\n", .{});
    warmupCaches(&test_hands, &simd_eval);

    // Measure overhead
    print("  Measuring framework overhead...\n", .{});
    const overhead_ns = benchmarkDummyEvaluator(iterations / 10, &test_hands);

    // Multiple benchmark runs
    const NUM_RUNS = 5;
    const use_purge = false; // Set to true if running with sudo
    var times: [NUM_RUNS]f64 = undefined;

    print("  Running {d} benchmark iterations...\n", .{NUM_RUNS});
    for (0..NUM_RUNS) |run| {
        if (run > 0) {
            clearCaches(use_purge);
            if (use_purge) {
                std.time.sleep(3_000_000_000); // 3 seconds
            }
        }

        times[run] = runSingleBenchmark(iterations, &test_hands, &simd_eval);
        print("    Run {d}: {d:.2} ns/hand\n", .{ run + 1, times[run] });
    }

    // Calculate statistics
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    const median = times[NUM_RUNS / 2];
    const net_median = median - overhead_ns;
    const cv = calculateCV(&times);

    const hands_per_sec = @as(u64, @intFromFloat(1_000_000_000.0 / net_median));

    // Results
    print("\n\x1b[1mBenchmark Results\x1b[0m\n", .{});
    print("  Iterations per run: {d} batches ({d} hands)\n", .{ iterations, iterations * BATCH_SIZE });
    print("  Framework overhead:   {d:.2} ns/hand\n", .{overhead_ns});
    print("  Raw median time:     {d:.2} ns/hand\n", .{median});
    print("  Net median time:     {d:.2} ns/hand\n", .{net_median});
    print("  Coefficient of var.: {d:.2}%\n", .{cv * 100});
    print("  Hands per second:    {d}\n", .{hands_per_sec});

    // Quality check
    if (cv > 0.05) {
        print("  ⚠️  High variation ({d:.2}%) - consider more stable environment\n", .{cv * 100});
    } else {
        print("  ✓ Low variation ({d:.2}%) - reliable measurement\n", .{cv * 100});
    }
    print("\n", .{});
}

// Single hand evaluation test for debugging
pub fn testSingleHand() !void {
    const print = std.debug.print;

    print("\x1b[1mSingle Hand Test\x1b[0m\n", .{});

    // Test a specific hand
    const test_hand: u64 = 0x1F00; // Royal flush clubs (A-K-Q-J-T of clubs)

    const slow_result = slow_evaluator.evaluateHand(test_hand);
    const fast_result = simd_evaluator.evaluate_single_hand(test_hand);

    print("Test hand:         0x{X}\n", .{test_hand});
    print("Slow evaluator:    {d}\n", .{slow_result});
    print("Fast evaluator:    {d}\n", .{fast_result});
    print("Match:             {s}\n", .{if (slow_result == fast_result) "✓" else "✗"});
    print("\n", .{});
}

pub fn main() !void {
    try testSingleHand();
    try testSimdEvaluator();
    try benchmarkSimdEvaluator(100000); // Reduced for 5-run methodology
}
