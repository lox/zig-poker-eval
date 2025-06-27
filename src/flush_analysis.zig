const std = @import("std");
const evaluator = @import("evaluator.zig");
const slow_evaluator = @import("slow_evaluator.zig");
const validation = @import("validation.zig");

const NUM_TEST_HANDS = 1_000_000;
const NUM_BENCHMARK_RUNS = 5;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n🎯 Flush vs Non-Flush Performance Analysis\n\n", .{});
    
    // Generate large set of hands
    const all_hands = try allocator.alloc(u64, NUM_TEST_HANDS);
    defer allocator.free(all_hands);
    
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    
    for (all_hands) |*hand| {
        hand.* = validation.generateRandomHand(&rng);
    }
    
    // Separate flush hands from non-flush hands
    var flush_hands = std.ArrayList(u64).init(allocator);
    var non_flush_hands = std.ArrayList(u64).init(allocator);
    defer flush_hands.deinit();
    defer non_flush_hands.deinit();
    
    std.debug.print("📊 Categorizing hands...\n", .{});
    for (all_hands) |hand| {
        if (isFlushHand(hand)) {
            try flush_hands.append(hand);
        } else {
            try non_flush_hands.append(hand);
        }
    }
    
    const flush_count = flush_hands.items.len;
    const non_flush_count = non_flush_hands.items.len;
    const flush_percentage = @as(f64, @floatFromInt(flush_count)) / @as(f64, @floatFromInt(NUM_TEST_HANDS)) * 100.0;
    
    std.debug.print("   Total hands: {}\n", .{NUM_TEST_HANDS});
    std.debug.print("   Flush hands: {} ({d:.2}%)\n", .{ flush_count, flush_percentage });
    std.debug.print("   Non-flush hands: {} ({d:.2}%)\n\n", .{ non_flush_count, 100.0 - flush_percentage });
    
    // Benchmark single-hand evaluation
    std.debug.print("🚀 Single-Hand Evaluation Benchmarks\n", .{});
    
    if (flush_count > 0) {
        const flush_time = try benchmarkSingleHands(flush_hands.items[0..@min(flush_count, 100000)]);
        std.debug.print("   Flush hands:     {d:.2} ns/hand\n", .{flush_time});
    }
    
    const non_flush_time = try benchmarkSingleHands(non_flush_hands.items[0..@min(non_flush_count, 100000)]);
    std.debug.print("   Non-flush hands: {d:.2} ns/hand\n\n", .{non_flush_time});
    
    // Benchmark SIMD evaluation
    std.debug.print("⚡ SIMD Evaluation Benchmarks\n", .{});
    
    if (flush_count > 0) {
        const flush_simd_time = try benchmarkSIMDHands(flush_hands.items[0..@min(flush_count, 100000)]);
        std.debug.print("   Flush hands:     {d:.2} ns/hand\n", .{flush_simd_time});
    }
    
    const non_flush_simd_time = try benchmarkSIMDHands(non_flush_hands.items[0..@min(non_flush_count, 100000)]);
    std.debug.print("   Non-flush hands: {d:.2} ns/hand\n\n", .{non_flush_simd_time});
    
    // Analyze the breakdown
    if (flush_count > 0) {
        const flush_time = try benchmarkSingleHands(flush_hands.items[0..@min(flush_count, 100000)]);
        const flush_simd_time = try benchmarkSIMDHands(flush_hands.items[0..@min(flush_count, 100000)]);
        
        std.debug.print("📈 Performance Analysis\n", .{});
        std.debug.print("   Flush performance:\n", .{});
        std.debug.print("     Single: {d:.2} ns/hand\n", .{flush_time});
        std.debug.print("     SIMD:   {d:.2} ns/hand ({d:.1}x speedup)\n", .{ flush_simd_time, flush_time / flush_simd_time });
        
        std.debug.print("   Non-flush performance:\n", .{});
        std.debug.print("     Single: {d:.2} ns/hand\n", .{non_flush_time});
        std.debug.print("     SIMD:   {d:.2} ns/hand ({d:.1}x speedup)\n\n", .{ non_flush_simd_time, non_flush_time / non_flush_simd_time });
        
        std.debug.print("💡 Insights:\n", .{});
        const flush_overhead = flush_simd_time - non_flush_simd_time;
        std.debug.print("   • Flush hands are {d:.2} ns slower than non-flush\n", .{flush_overhead});
        std.debug.print("   • Flush detection adds {d:.1}% overhead\n", .{ flush_overhead / non_flush_simd_time * 100.0 });
        
        // Calculate weighted average impact
        const weighted_impact = flush_overhead * (flush_percentage / 100.0);
        std.debug.print("   • Weighted impact on average performance: {d:.3} ns/hand\n", .{weighted_impact});
    } else {
        std.debug.print("⚠️  No flush hands found in sample - increase sample size or adjust generation\n", .{});
    }
}

fn isFlushHand(hand: u64) bool {
    // Check each suit for 5+ cards
    const suits = slow_evaluator.getSuitMasks(hand);
    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            return true;
        }
    }
    return false;
}

fn benchmarkSingleHands(hands: []const u64) !f64 {
    var times: [NUM_BENCHMARK_RUNS]f64 = undefined;
    
    for (&times) |*time| {
        var checksum: u64 = 0;
        const start = std.time.nanoTimestamp();
        
        for (hands) |hand| {
            checksum +%= evaluator.evaluate_hand(hand);
        }
        
        const end = std.time.nanoTimestamp();
        std.mem.doNotOptimizeAway(checksum);
        
        time.* = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(hands.len));
    }
    
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[NUM_BENCHMARK_RUNS / 2]; // median
}

fn benchmarkSIMDHands(hands: []const u64) !f64 {
    const batch_size = 4;
    
    // Pad hands to batch boundary
    const padded_count = (hands.len / batch_size) * batch_size;
    if (padded_count == 0) return 0.0;
    
    var times: [NUM_BENCHMARK_RUNS]f64 = undefined;
    
    for (&times) |*time| {
        var checksum: u64 = 0;
        const start = std.time.nanoTimestamp();
        
        var i: usize = 0;
        while (i < padded_count) {
            // Create batch
            var batch_hands: [4]u64 = undefined;
            for (0..batch_size) |j| {
                batch_hands[j] = hands[i + j];
            }
            const batch = @as(@Vector(4, u64), batch_hands);
            
            const results = evaluator.evaluate_batch_4(batch);
            for (0..batch_size) |j| {
                checksum +%= results[j];
            }
            
            i += batch_size;
        }
        
        const end = std.time.nanoTimestamp();
        std.mem.doNotOptimizeAway(checksum);
        
        time.* = @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(padded_count));
    }
    
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    return times[NUM_BENCHMARK_RUNS / 2]; // median
}