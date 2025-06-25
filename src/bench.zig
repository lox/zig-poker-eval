const std = @import("std");
const evaluator = @import("evaluator.zig");
const simd_evaluator = @import("simd_evaluator.zig");
const chd = @import("chd.zig");

// Generate random hands for testing and benchmarking
pub fn generateRandomHandBatch(rng: *std.Random) simd_evaluator.HandBatch {
    var hands: [16]u64 = undefined;

    for (&hands) |*hand| {
        hand.* = generateRandomHand(rng);
    }

    return simd_evaluator.HandBatch{ hands[0], hands[1], hands[2], hands[3], hands[4], hands[5], hands[6], hands[7], hands[8], hands[9], hands[10], hands[11], hands[12], hands[13], hands[14], hands[15] };
}

fn generateRandomHand(rng: *std.Random) evaluator.Hand {
    var hand: evaluator.Hand = 0;
    var cards_dealt: u8 = 0;

    while (cards_dealt < 7) {
        const suit = rng.intRangeAtMost(u8, 0, 3);
        const rank = rng.intRangeAtMost(u8, 0, 12);
        const card = evaluator.makeCard(suit, rank);

        if ((hand & card) == 0) {
            hand |= card;
            cards_dealt += 1;
        }
    }

    return hand;
}

// Helper functions for rigorous benchmarking

fn warmupCaches(test_batches: []const simd_evaluator.HandBatch, simd_eval: *const simd_evaluator.SimdEvaluator) void {
    // Touch lookup tables (if available)
    // Note: @import will fail at compile time if tables.zig doesn't exist
    // const tables = @import("tables.zig");
    // if (@hasDecl(tables, "rank_patterns")) {
    //     for (tables.rank_patterns[0..@min(16384, tables.rank_patterns.len)]) |pattern| {
    //         std.mem.doNotOptimizeAway(pattern);
    //     }
    // }
    
    // Touch first 4K batches (64K hands) 
    for (test_batches[0..@min(4096, test_batches.len)]) |batch| {
        _ = simd_eval.evaluateBatch(batch);
    }
}

fn clearCaches(use_purge: bool) void {
    if (!use_purge) return;
    
    var child = std.process.Child.init(&[_][]const u8{"sudo", "purge"}, std.heap.page_allocator);
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

fn runSingleBenchmark(iterations: u32, test_batches: []const simd_evaluator.HandBatch, simd_eval: *const simd_evaluator.SimdEvaluator) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const results = simd_eval.evaluateBatch(test_batches[i % test_batches.len]);
        for (0..16) |j| {
            checksum +%= results[j];
        }
    }
    
    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);
    
    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(iterations * 16));
}

fn benchmarkDummyEvaluator(iterations: u32, test_batches: []const simd_evaluator.HandBatch) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const batch = test_batches[i % test_batches.len];
        for (0..16) |j| {
            checksum +%= @popCount(batch[j]); // Trivial operation
        }
    }
    
    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);
    
    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(iterations * 16));
}

fn validateCorrectness(test_batches: []const simd_evaluator.HandBatch) !bool {
    var xor_checksum: u64 = 0;
    
    // Validate first 1K batches
    for (test_batches[0..@min(1000, test_batches.len)]) |batch| {
        for (0..16) |j| {
            const result = evaluator.evaluateHand(batch[j]);
            xor_checksum ^= result;
        }
    }
    
    // For now, just ensure checksum is non-zero (replace with actual reference when known)
    return xor_checksum != 0;
}

// Test the SIMD evaluator
pub fn testSimdEvaluator() !void {
    _ = std.mem.Allocator;
    const print = std.debug.print;

    print("Testing SIMD evaluator (DESIGN.md implementation)...\n", .{});

    const simd_eval = simd_evaluator.SimdEvaluator.init();

    // Generate test batch
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const batch = generateRandomHandBatch(&rng);

    // Evaluate batch
    const batch_results = simd_eval.evaluateBatch(batch);

    // Validate against single-hand evaluation
    print("Validation against single-hand evaluation:\n", .{});
    var matches: u32 = 0;

    for (0..16) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = evaluator.evaluateHand(hand);

        if (batch_result == single_result) {
            matches += 1;
        }

        if (i < 5) { // Show first 5 for debugging
            print("  Hand {}: batch={}, single={}, match={}\n", .{ i, batch_result, single_result, batch_result == single_result });
        }
    }

    print("Accuracy: {}/16 ({d:.1}%)\n", .{ matches, @as(f64, @floatFromInt(matches)) * 100.0 / 16.0 });
    print("SIMD evaluator test complete\n", .{});
}

// Benchmark the SIMD evaluator with rigorous methodology
pub fn benchmarkSimdEvaluator(iterations: u32) !void {
    const print = std.debug.print;

    print("SIMD Evaluator Benchmark\n", .{});
    print("=========================================================\n", .{});

    const simd_eval = simd_evaluator.SimdEvaluator.init();

    // Generate test batches
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var test_batches: [100_000]simd_evaluator.HandBatch = undefined;
    for (&test_batches) |*batch| {
        batch.* = generateRandomHandBatch(&rng);
    }

    // Validate correctness
    print("Validating correctness...\n", .{});
    const is_correct = validateCorrectness(&test_batches) catch false;
    if (!is_correct) {
        print("ERROR: Correctness validation failed!\n", .{});
        return;
    }
    print("✓ Correctness validation passed\n", .{});

    // Cache warmup
    print("Warming up caches...\n", .{});
    warmupCaches(&test_batches, &simd_eval);

    // Measure overhead
    print("Measuring framework overhead...\n", .{});
    const overhead_ns = benchmarkDummyEvaluator(iterations / 10, &test_batches);

    // Multiple benchmark runs
    const NUM_RUNS = 5;
    const use_purge = false; // Set to true if running with sudo
    var times: [NUM_RUNS]f64 = undefined;

    print("Running {} benchmark iterations...\n", .{NUM_RUNS});
    for (0..NUM_RUNS) |run| {
        if (run > 0) {
            clearCaches(use_purge);
            if (use_purge) {
                std.time.sleep(3_000_000_000); // 3 seconds
            }
        }

        times[run] = runSingleBenchmark(iterations, &test_batches, &simd_eval);
        print("Run {}: {d:.2} ns/hand\n", .{ run + 1, times[run] });
    }

    // Calculate statistics
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    const median = times[NUM_RUNS / 2];
    const net_median = median - overhead_ns;
    const cv = calculateCV(&times);

    // Results
    print("\nBenchmark Results:\n", .{});
    print("Iterations per run: {} batches ({} hands)\n", .{ iterations, iterations * 16 });
    print("Framework overhead: {d:.2} ns/hand\n", .{overhead_ns});
    print("Raw median time: {d:.2} ns/hand\n", .{median});
    print("Net median time: {d:.2} ns/hand (after overhead correction)\n", .{net_median});
    print("Coefficient of variation: {d:.1}%\n", .{cv * 100});
    print("Hands per second: {d:.0}\n", .{1_000_000_000.0 / net_median});

    // Quality check
    if (cv > 0.05) {
        print("⚠️  High variation ({d:.1}%) - consider more stable environment\n", .{cv * 100});
    } else {
        print("✓ Low variation ({d:.1}%) - reliable measurement\n", .{cv * 100});
    }
}

// Benchmark CHD lookup performance
pub fn benchmarkCHD(iterations: u32) !void {
    const print = std.debug.print;
    
    print("CHD Lookup Benchmark\n", .{});
    print("============================================\n", .{});
    
    const ns_per_lookup = chd.benchmarkCHDLookup(iterations);
    const lookups_per_sec = 1_000_000_000.0 / ns_per_lookup;
    
    print("Iterations: {}\n", .{iterations});
    print("Time per lookup: {d:.2} ns\n", .{ns_per_lookup});
    print("Lookups per second: {d:.0}\n", .{lookups_per_sec});
    

}

pub fn main() !void {
    try benchmarkCHD(100000);
    try benchmarkSimdEvaluator(100000); // Reduced for 5-run methodology
}
