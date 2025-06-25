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

// Benchmark the SIMD evaluator (target: 2-5 ns/hand)
pub fn benchmarkSimdEvaluator(iterations: u32) !void {
    _ = std.mem.Allocator;
    const print = std.debug.print;

    print("SIMD Evaluator Benchmark (DESIGN.md target: 2-5 ns/hand)\n", .{});
    print("=========================================================\n", .{});

    // Tables are now pre-compiled in tables.zig

    const simd_eval = simd_evaluator.SimdEvaluator.init();

    // Generate test batches
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var test_batches: [100]simd_evaluator.HandBatch = undefined;
    for (&test_batches) |*batch| {
        batch.* = generateRandomHandBatch(&rng);
    }

    // Warmup
    for (0..100) |i| {
        _ = simd_eval.evaluateBatch(test_batches[i % test_batches.len]);
    }

    // Measure performance
    var checksum: u64 = 0; // Prevent optimization
    const start = std.time.nanoTimestamp();

    for (0..iterations) |i| {
        const results = simd_eval.evaluateBatch(test_batches[i % test_batches.len]);
        // Sum results to prevent optimization
        for (0..16) |j| {
            checksum +%= results[j];
        }
    }

    const end = std.time.nanoTimestamp();

    // Use checksum to prevent dead code elimination
    if (checksum == 0) {
        print("Warning: unexpected zero checksum\n", .{});
    }

    // Calculate metrics
    const total_ns = @as(f64, @floatFromInt(end - start));
    const total_hands = @as(f64, @floatFromInt(iterations * 16));
    const ns_per_hand = total_ns / total_hands;
    const ns_per_batch = total_ns / @as(f64, @floatFromInt(iterations));
    const hands_per_sec = 1_000_000_000.0 / ns_per_hand;

    // Results
    print("Iterations: {} batches ({} hands)\n", .{ iterations, iterations * 16 });
    print("Total time: {d:.2} ms\n", .{total_ns / 1_000_000.0});
    print("Time per batch: {d:.2} ns\n", .{ns_per_batch});
    print("Time per hand: {d:.2} ns\n", .{ns_per_hand});
    print("Hands per second: {d:.0}\n", .{hands_per_sec});
    print("Checksum: {} (prevents optimization)\n", .{checksum});

    // Compare with DESIGN.md targets
    print("\nDESIGN.md Performance Targets:\n", .{});
    print("  Hot L1, random hands: 2.0-2.4 ns/hand\n", .{});
    print("  Stressed L2 (multi-thread): 2.9 ns/hand\n", .{});
    print("  AVX2 (8-lane) fallback: 5.0 ns/hand\n", .{});

    if (ns_per_hand <= 2.4) {
        print("  Status: 🎯 TARGET ACHIEVED! (Hot L1 performance)\n", .{});
    } else if (ns_per_hand <= 2.9) {
        print("  Status: 🔥 EXCELLENT! (Stressed L2 performance)\n", .{});
    } else if (ns_per_hand <= 5.0) {
        print("  Status: ✅ GOOD! (AVX2 fallback performance)\n", .{});
    } else if (ns_per_hand <= 10.0) {
        print("  Status: 🔄 Close ({d:.1}x slower than target)\n", .{ns_per_hand / 5.0});
    } else {
        print("  Status: ⚠️  Needs optimization ({d:.1}x slower than target)\n", .{ns_per_hand / 5.0});
    }

    print("\n", .{});
}

// Benchmark CHD lookup performance
pub fn benchmarkCHD(iterations: u32) !void {
    const print = std.debug.print;
    
    print("CHD Lookup Benchmark (DESIGN.md RPC tables)\n", .{});
    print("============================================\n", .{});
    
    const ns_per_lookup = chd.benchmarkCHDLookup(iterations);
    const lookups_per_sec = 1_000_000_000.0 / ns_per_lookup;
    
    print("Iterations: {}\n", .{iterations});
    print("Time per lookup: {d:.2} ns\n", .{ns_per_lookup});
    print("Lookups per second: {d:.0}\n", .{lookups_per_sec});
    
    print("\nCHD vs DESIGN.md targets:\n", .{});
    if (ns_per_lookup <= 2.5) {
        print("  Status: 🎯 EXCELLENT! (Within 2.5ns target)\n", .{});
    } else if (ns_per_lookup <= 5.0) {
        print("  Status: ✅ GOOD! (Within 5ns target)\n", .{});
    } else {
        print("  Status: ⚠️  Needs optimization ({d:.1}x slower than 2.5ns target)\n", .{ns_per_lookup / 2.5});
    }
    
    print("\n", .{});
}

pub fn main() !void {
    try benchmarkCHD(100000);
    try benchmarkSimdEvaluator(1000000);
}
