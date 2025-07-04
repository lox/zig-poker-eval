const std = @import("std");
const evaluator = @import("evaluator");

pub const BenchmarkOptions = struct {
    iterations: u32 = 100000,
    warmup: bool = true,
    measure_overhead: bool = true,
    multiple_runs: bool = true,
    show_comparison: bool = true,
    verbose: bool = false,
};

pub const BenchmarkResult = struct {
    batch_ns_per_hand: f64,
    single_ns_per_hand: f64,
    simd_speedup: f64,
    hands_per_second: u64,
    coefficient_variation: f64,
    overhead_ns: f64,
    total_hands: u64,
};

// Helper functions for rigorous benchmarking
const BATCH_SIZE = 4;

// Helper to create batches from hand arrays
fn createBatch(hands: []const u64, start_idx: usize) @Vector(BATCH_SIZE, u64) {
    var batch_hands: [BATCH_SIZE]u64 = undefined;
    for (0..BATCH_SIZE) |i| {
        batch_hands[i] = hands[(start_idx + i) % hands.len];
    }
    return @as(@Vector(BATCH_SIZE, u64), batch_hands);
}

fn warmupCaches(test_hands: []const u64) void {
    // Touch lookup tables by performing lookups - access through public evaluator API
    var prng = std.Random.DefaultPrng.init(123);
    var rng = prng.random();
    
    // Warm up by evaluating some random hands
    for (0..1024) |_| {
        const hand = evaluator.generateRandomHand(&rng);
        std.mem.doNotOptimizeAway(evaluator.evaluateHand(hand));
    }

    // Touch first portion of hands by evaluating them in batches
    const warmup_hands = @min(65536, test_hands.len); // 64K hands max
    var i: usize = 0;
    while (i + BATCH_SIZE <= warmup_hands) {
        const batch = createBatch(test_hands, i);
        _ = evaluator.evaluateBatch4(batch);
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

fn runSingleBenchmark(iterations: u32, test_hands: []const u64) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    const total_hands = iterations * BATCH_SIZE;

    var hand_idx: usize = 0;
    for (0..iterations) |_| {
        // Create batch from consecutive hands
        const batch = createBatch(test_hands, hand_idx);
        
        const results = evaluator.evaluateBatch4(batch);
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

fn benchmarkSingleHand(test_hands: []const u64, count: u32) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..count) |i| {
        checksum +%= evaluator.evaluateHand(test_hands[i % test_hands.len]);
    }
    
    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);
    
    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(count));
}

pub fn runBenchmark(options: BenchmarkOptions, allocator: std.mem.Allocator) !BenchmarkResult {
    // Generate test hands
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const num_test_hands = 1_600_000; // Large pool for cache effects
    const test_hands = try allocator.alloc(u64, num_test_hands);
    defer allocator.free(test_hands);
    
    for (test_hands) |*hand| {
        hand.* = evaluator.generateRandomHand(&rng);
    }

    var result = BenchmarkResult{
        .batch_ns_per_hand = 0,
        .single_ns_per_hand = 0,
        .simd_speedup = 0,
        .hands_per_second = 0,
        .coefficient_variation = 0,
        .overhead_ns = 0,
        .total_hands = 0,
    };

    // Cache warmup
    if (options.warmup) {
        warmupCaches(test_hands);
    }

    // Measure overhead
    if (options.measure_overhead) {
        result.overhead_ns = benchmarkDummyEvaluator(options.iterations / 10, test_hands);
    }

    // Batch benchmark
    if (options.multiple_runs) {
        // Multiple runs for statistical analysis
        const NUM_RUNS = 5;
        const use_purge = false; // Set to true if running with sudo
        var times: [NUM_RUNS]f64 = undefined;

        for (0..NUM_RUNS) |run| {
            if (run > 0) {
                clearCaches(use_purge);
                if (use_purge) {
                    std.time.sleep(3_000_000_000); // 3 seconds
                }
            }

            times[run] = runSingleBenchmark(options.iterations, test_hands);
        }

        // Calculate statistics
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        const median = times[NUM_RUNS / 2];
        result.batch_ns_per_hand = median - result.overhead_ns;
        result.coefficient_variation = calculateCV(&times);
    } else {
        // Single run
        const raw_time = runSingleBenchmark(options.iterations, test_hands);
        result.batch_ns_per_hand = raw_time - result.overhead_ns;
        result.coefficient_variation = 0.0;
    }

    // Single-hand benchmark for comparison
    if (options.show_comparison) {
        result.single_ns_per_hand = benchmarkSingleHand(test_hands, 10000);
        result.simd_speedup = result.single_ns_per_hand / result.batch_ns_per_hand;
    }

    result.hands_per_second = @as(u64, @intFromFloat(1_000_000_000.0 / result.batch_ns_per_hand));
    result.total_hands = options.iterations * BATCH_SIZE;

    return result;
}

pub fn validateCorrectness(test_hands: []const u64) !bool {
    var matches: u32 = 0;
    var total: u32 = 0;

    // Validate first 16K hands in batches
    const validation_hands = @min(16000, test_hands.len);
    var i: usize = 0;
    while (i + BATCH_SIZE <= validation_hands) {
        const batch = createBatch(test_hands, i);
        
        const fast_results = evaluator.evaluateBatch4(batch);

        for (0..BATCH_SIZE) |j| {
            const slow_result = evaluator.slow.evaluateHand(test_hands[i + j]);
            const fast_result = fast_results[j];

            if (slow_result == fast_result) {
                matches += 1;
            }
            total += 1;
        }
        i += BATCH_SIZE;
    }

    const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total));
    
    // Require 100% accuracy
    if (accuracy < 1.0) {
        return error.AccuracyTooLow;
    }
    
    return true;
}

// Test the evaluator with a specific test batch
pub fn testEvaluator() !void {
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = evaluator.generateRandomHandBatch(&rng);

    // Evaluate batch
    const batch_results = evaluator.evaluateBatch4(batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    for (0..BATCH_SIZE) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = evaluator.slow.evaluateHand(hand);

        if (batch_result == single_result) {
            matches += 1;
        }
    }

    if (matches != BATCH_SIZE) {
        return error.EvaluatorMismatch;
    }
}

// Test a specific hand for debugging
pub fn testSingleHand(hand: u64) struct { slow: u16, fast: u16, match: bool } {
    const slow_result = evaluator.slow.evaluateHand(hand);
    const fast_result = evaluator.evaluateHand(hand);
    
    return .{
        .slow = slow_result,
        .fast = fast_result,
        .match = slow_result == fast_result,
    };
}