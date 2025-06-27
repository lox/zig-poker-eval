const std = @import("std");
const tables = @import("tables.zig");
const slow_evaluator = @import("slow_evaluator.zig");

// Simple, clean evaluator - no premature optimization
const RANK_MASK = 0x1FFF; // 13 bits for ranks

// === Core Algorithm (from your DESIGN.md) ===

fn compute_rpc_from_hand(hand: u64) u32 {
    var rank_counts = [_]u8{0} ** 13;

    for (0..4) |suit| {
        const suit_mask = @as(u16, @truncate((hand >> (@as(u6, @intCast(suit)) * 13)) & RANK_MASK));
        for (0..13) |rank| {
            if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    // Base-5 encoding: preserves all patterns in 31 bits
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return rpc;
}

fn mix64(x: u64) u64 {
    var result = x;
    result ^= result >> 33;
    result *%= tables.CHD_MAGIC_CONSTANT;
    result ^= result >> 29;
    return result;
}

fn chd_lookup_scalar(rpc: u32) u16 {
    const h = mix64(@as(u64, rpc));
    const bucket = @as(u32, @intCast(h >> 51)); // Top 13 bits
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits
    const displacement = tables.chd_g_array[bucket];
    const final_index = (base_index + displacement) & (tables.CHD_TABLE_SIZE - 1);
    return tables.chd_value_table[final_index];
}

fn is_flush_hand(hand: u64) bool {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK,   // clubs
        @as(u16, @truncate(hand >> 13)) & RANK_MASK,  // diamonds
        @as(u16, @truncate(hand >> 26)) & RANK_MASK,  // hearts
        @as(u16, @truncate(hand >> 39)) & RANK_MASK,  // spades
    };
    
    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) return true;
    }
    return false;
}

// === Public API ===

pub fn evaluate_hand(hand: u64) u16 {
    if (is_flush_hand(hand)) {
        return slow_evaluator.evaluateHand(hand); // Handle flushes properly
    }
    
    const rpc = compute_rpc_from_hand(hand);
    return chd_lookup_scalar(rpc);
}

// Batch evaluation - let the compiler vectorize
pub fn evaluate_batch_4(hands: @Vector(4, u64)) @Vector(4, u16) {
    var results: @Vector(4, u16) = @splat(0);
    
    inline for (0..4) |i| {
        results[i] = evaluate_hand(hands[i]);
    }
    
    return results;
}

// Architecture-adaptive batch sizes (simple approach)
pub fn evaluate_batch_dynamic(hands: []const u64, results: []u16) void {
    std.debug.assert(hands.len == results.len);
    
    // Process in chunks of 4 (optimal for most architectures)
    var i: usize = 0;
    while (i + 4 <= hands.len) : (i += 4) {
        const batch_hands = @Vector(4, u64){ hands[i], hands[i+1], hands[i+2], hands[i+3] };
        const batch_results = evaluate_batch_4(batch_hands);
        
        results[i] = batch_results[0];
        results[i+1] = batch_results[1];
        results[i+2] = batch_results[2];
        results[i+3] = batch_results[3];
    }
    
    // Handle remainder
    while (i < hands.len) : (i += 1) {
        results[i] = evaluate_hand(hands[i]);
    }
}

// === Testing/Benchmarking ===

pub fn benchmark_single(iterations: u32) u64 {
    var sum: u64 = 0;
    const test_hand: u64 = 0x123456789ABCD;
    
    for (0..iterations) |_| {
        sum +%= evaluate_hand(test_hand);
    }
    
    return sum;
}

pub fn benchmark_batch(iterations: u32) u64 {
    var sum: u64 = 0;
    const test_hands = @Vector(4, u64){ 
        0x1F00000000000, 0x123456789ABCD, 0x0F0F0F0F0F0F0, 0x1F00 
    };
    
    for (0..iterations) |_| {
        const results = evaluate_batch_4(test_hands);
        sum +%= results[0] + results[1] + results[2] + results[3];
    }
    
    return sum;
}