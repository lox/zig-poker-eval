const std = @import("std");
const poker = @import("poker.zig");
const benchmark = @import("benchmark.zig");

// Detailed rank extraction micro-profiling
pub fn profileRankExtractionDetails(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("=== DETAILED RANK EXTRACTION ANALYSIS ===\n", .{});
    
    const hands = try benchmark.generateRandomHands(allocator, 100_000, 42);
    defer allocator.free(hands);
    
    const iterations = 1_000_000;
    print("Testing {} iterations on {} hands...\n", .{iterations, hands.len});
    
    // Test current approach
    var dummy_result: u64 = 0;
    const start_current = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const result = poker.extractRankDataOptimized(hand.bits);
        dummy_result += result.mask;
    }
    const end_current = std.time.nanoTimestamp();
    
    // Test manual unrolled approach
    const start_unrolled = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const result = extractRankDataUnrolled(hand.bits);
        dummy_result += result.mask;
    }
    const end_unrolled = std.time.nanoTimestamp();
    
    // Test bit manipulation approach
    const start_bitmask = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const result = extractRankDataBitManipulation(hand.bits);
        dummy_result += result.mask;
    }
    const end_bitmask = std.time.nanoTimestamp();
    
    // Test NEON approach (if available)
    const start_neon = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const result = extractRankDataNEON(hand.bits);
        dummy_result += result.mask;
    }
    const end_neon = std.time.nanoTimestamp();
    
    const time_current = end_current - start_current;
    const time_unrolled = end_unrolled - start_unrolled;
    const time_bitmask = end_bitmask - start_bitmask;
    const time_neon = end_neon - start_neon;
    
    print("\nResults (ns per call):\n", .{});
    print("  Current approach:     {d:.2}ns\n", .{@as(f64, @floatFromInt(time_current)) / iterations});
    print("  Manual unrolled:      {d:.2}ns\n", .{@as(f64, @floatFromInt(time_unrolled)) / iterations});
    print("  Bit manipulation:     {d:.2}ns\n", .{@as(f64, @floatFromInt(time_bitmask)) / iterations});
    print("  NEON parallel:        {d:.2}ns\n", .{@as(f64, @floatFromInt(time_neon)) / iterations});
    print("  Dummy result: {} (prevents optimization)\n", .{dummy_result});
    
    // Test correctness
    print("\nCorrectness verification...\n", .{});
    const test_hand = hands[0];
    const result_current = poker.extractRankDataOptimized(test_hand.bits);
    const result_unrolled = extractRankDataUnrolled(test_hand.bits);
    const result_bitmask = extractRankDataBitManipulation(test_hand.bits);
    const result_neon = extractRankDataNEON(test_hand.bits);
    
    const current_match = std.mem.eql(u8, &result_current.counts, &result_unrolled.counts) and result_current.mask == result_unrolled.mask;
    const bitmask_match = std.mem.eql(u8, &result_current.counts, &result_bitmask.counts) and result_current.mask == result_bitmask.mask;
    const neon_match = std.mem.eql(u8, &result_current.counts, &result_neon.counts) and result_current.mask == result_neon.mask;
    
    print("  Unrolled matches current: {}\n", .{current_match});
    print("  Bit manipulation matches: {}\n", .{bitmask_match});
    print("  NEON matches current:     {}\n", .{neon_match});
}

// Manual unrolled approach - let compiler see all operations
inline fn extractRankDataUnrolled(hand_bits: u64) struct { counts: [13]u8, mask: u16 } {
    // Unroll all 13 rank extractions for maximum parallelization
    const r0 = @popCount((hand_bits >> 0) & 0xF);   // Rank 2
    const r1 = @popCount((hand_bits >> 4) & 0xF);   // Rank 3
    const r2 = @popCount((hand_bits >> 8) & 0xF);   // Rank 4
    const r3 = @popCount((hand_bits >> 12) & 0xF);  // Rank 5
    const r4 = @popCount((hand_bits >> 16) & 0xF);  // Rank 6
    const r5 = @popCount((hand_bits >> 20) & 0xF);  // Rank 7
    const r6 = @popCount((hand_bits >> 24) & 0xF);  // Rank 8
    const r7 = @popCount((hand_bits >> 28) & 0xF);  // Rank 9
    const r8 = @popCount((hand_bits >> 32) & 0xF);  // Rank T
    const r9 = @popCount((hand_bits >> 36) & 0xF);  // Rank J
    const r10 = @popCount((hand_bits >> 40) & 0xF); // Rank Q
    const r11 = @popCount((hand_bits >> 44) & 0xF); // Rank K
    const r12 = @popCount((hand_bits >> 48) & 0xF); // Rank A
    
    // Build rank mask in parallel
    const mask = 
        (@as(u16, @intFromBool(r0 > 0)) << 0) |
        (@as(u16, @intFromBool(r1 > 0)) << 1) |
        (@as(u16, @intFromBool(r2 > 0)) << 2) |
        (@as(u16, @intFromBool(r3 > 0)) << 3) |
        (@as(u16, @intFromBool(r4 > 0)) << 4) |
        (@as(u16, @intFromBool(r5 > 0)) << 5) |
        (@as(u16, @intFromBool(r6 > 0)) << 6) |
        (@as(u16, @intFromBool(r7 > 0)) << 7) |
        (@as(u16, @intFromBool(r8 > 0)) << 8) |
        (@as(u16, @intFromBool(r9 > 0)) << 9) |
        (@as(u16, @intFromBool(r10 > 0)) << 10) |
        (@as(u16, @intFromBool(r11 > 0)) << 11) |
        (@as(u16, @intFromBool(r12 > 0)) << 12);
    
    return .{
        .counts = [13]u8{ r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 },
        .mask = mask,
    };
}

// Bit manipulation approach - extract patterns without individual popcounts
inline fn extractRankDataBitManipulation(hand_bits: u64) struct { counts: [13]u8, mask: u16 } {
    var rank_counts: [13]u8 = undefined;
    var rank_mask: u16 = 0;
    
    // Use bit manipulation tricks to process multiple ranks simultaneously
    // (Note: suit extraction not used in this approach - kept for future SIMD development)
    _ = hand_bits & 0x1111111111111111; // Hearts
    _ = hand_bits & 0x2222222222222222; // Spades
    _ = hand_bits & 0x4444444444444444; // Diamonds  
    _ = hand_bits & 0x8888888888888888; // Clubs
    
    // Build rank presence mask and counts in one pass
    inline for (0..13) |rank_idx| {
        const rank_shift = rank_idx * 4;
        const rank_bits = (hand_bits >> rank_shift) & 0xF;
        const count = @popCount(rank_bits);
        rank_counts[rank_idx] = count;
        if (count > 0) {
            rank_mask |= @as(u16, 1) << @intCast(rank_idx);
        }
    }
    
    return .{
        .counts = rank_counts,
        .mask = rank_mask,
    };
}

// NEON approach - use ARM64 SIMD for parallel processing
inline fn extractRankDataNEON(hand_bits: u64) struct { counts: [13]u8, mask: u16 } {
    // For now, implement a more efficient bit manipulation approach
    // True NEON would require inline assembly or compiler intrinsics
    
    // Parallel rank extraction using vector-like operations
    var rank_counts: [13]u8 = undefined;
    var rank_mask: u16 = 0;
    
    // Process ranks in groups for better parallelization
    // Group 1: ranks 0-3
    const group1 = hand_bits & 0x000000000000FFFF;
    rank_counts[0] = @popCount((group1 >> 0) & 0xF);
    rank_counts[1] = @popCount((group1 >> 4) & 0xF);
    rank_counts[2] = @popCount((group1 >> 8) & 0xF);
    rank_counts[3] = @popCount((group1 >> 12) & 0xF);
    
    // Group 2: ranks 4-7
    const group2 = (hand_bits >> 16) & 0x000000000000FFFF;
    rank_counts[4] = @popCount((group2 >> 0) & 0xF);
    rank_counts[5] = @popCount((group2 >> 4) & 0xF);
    rank_counts[6] = @popCount((group2 >> 8) & 0xF);
    rank_counts[7] = @popCount((group2 >> 12) & 0xF);
    
    // Group 3: ranks 8-11
    const group3 = (hand_bits >> 32) & 0x000000000000FFFF;
    rank_counts[8] = @popCount((group3 >> 0) & 0xF);
    rank_counts[9] = @popCount((group3 >> 4) & 0xF);
    rank_counts[10] = @popCount((group3 >> 8) & 0xF);
    rank_counts[11] = @popCount((group3 >> 12) & 0xF);
    
    // Group 4: rank 12 (ace)
    rank_counts[12] = @popCount((hand_bits >> 48) & 0xF);
    
    // Build rank mask efficiently
    inline for (0..13) |i| {
        if (rank_counts[i] > 0) {
            rank_mask |= @as(u16, 1) << @intCast(i);
        }
    }
    
    return .{
        .counts = rank_counts,
        .mask = rank_mask,
    };
}