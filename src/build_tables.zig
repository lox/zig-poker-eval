const std = @import("std");

// Hand strength values (matching ultra-fast evaluator)
const HandStrength = enum(u16) {
    high_card = 0,
    pair = 1000,
    two_pair = 2000,
    three_kind = 3000,
    straight = 4000,
    flush = 5000,
    full_house = 6000,
    four_kind = 7000,
    straight_flush = 8000,
    royal_flush = 9000,
};

// Table sizes for L1 cache optimization
const MINI_HASH_SIZE = 2048; // 11-bit keys
const FLUSH_TABLE_SIZE = 1024; // 10-bit keys

// Bit manipulation constants (matching ultra-fast evaluator)
const FLUSH_CHECK_MASK: u64 = 0x1111_1111_1111_1111;
const RANK_EXTRACT_MASK: u64 = 0x000F_000F_000F_000F;

const HandTypeCounts = struct {
    quad_count: u32 = 0,
    full_count: u32 = 0,
    straight_count: u32 = 0,
    trips_count: u32 = 0,
    two_pair_count: u32 = 0,
    one_pair_count: u32 = 0,
    high_card_count: u32 = 0,
    flush_count: u32 = 0,
    straight_flush_count: u32 = 0,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const A = gpa.allocator();

    std.debug.print("Building Ultra-Fast L1-Optimized Tables\n", .{});
    std.debug.print("======================================\n", .{});

    // Initialize tables
    var mini_hash_table: [MINI_HASH_SIZE]u16 = [_]u16{0} ** MINI_HASH_SIZE;
    var flush_table: [FLUSH_TABLE_SIZE]u16 = [_]u16{0} ** FLUSH_TABLE_SIZE;
    var hand_counts = HandTypeCounts{};

    // Build flush lookup table first
    try buildFlushTable(&flush_table, &hand_counts);

    // Build non-flush lookup table
    try buildMiniHashTable(&mini_hash_table, &hand_counts, A);

    // Write tables to file
    try writeTables(&mini_hash_table, &flush_table);

    // Print statistics
    std.debug.print("✓ Built ultra-fast tables successfully\n", .{});
    std.debug.print("  Mini hash table: {} entries ({}KB)\n", .{ MINI_HASH_SIZE, MINI_HASH_SIZE * 2 / 1024 });
    std.debug.print("  Flush table: {} entries ({}KB)\n", .{ FLUSH_TABLE_SIZE, FLUSH_TABLE_SIZE * 2 / 1024 });
    std.debug.print("  Total size: {}KB (fits in L1 cache)\n", .{(MINI_HASH_SIZE + FLUSH_TABLE_SIZE) * 2 / 1024});

    printHandCounts(&hand_counts);
}

// Build flush table for all possible flush combinations
fn buildFlushTable(flush_table: *[FLUSH_TABLE_SIZE]u16, hand_counts: *HandTypeCounts) !void {
    std.debug.print("Building flush table...\n", .{});

    // Enumerate all possible 13-bit rank combinations for flushes
    for (0..8192) |rank_mask_full| {
        const rank_mask = @as(u16, @intCast(rank_mask_full));
        const rank_count = @popCount(rank_mask);

        // Need at least 5 cards for a flush
        if (rank_count < 5) continue;

        // Compress to 10-bit key
        const compressed_key = compressFlushKey(rank_mask);
        if (compressed_key >= FLUSH_TABLE_SIZE) continue;

        // Evaluate this flush combination
        const strength = evaluateFlush(rank_mask, hand_counts);

        // Store in table (handle collisions by keeping higher strength)
        if (flush_table[compressed_key] == 0 or strength > flush_table[compressed_key]) {
            flush_table[compressed_key] = strength;
        }
    }
}

// Build mini hash table for non-flush hands
fn buildMiniHashTable(mini_table: *[MINI_HASH_SIZE]u16, hand_counts: *HandTypeCounts, alloc: std.mem.Allocator) !void {
    std.debug.print("Building mini hash table...\n", .{});

    // Use collision resolution map
    var collision_map = std.AutoHashMap(u16, u16).init(alloc);
    defer collision_map.deinit();

    // Enumerate all 7-card combinations (reuse existing logic)
    var counts: [13]u8 = undefined;
    var total_count: u32 = 0;
    try enumerateRankCombinations(mini_table, &collision_map, &counts, 0, 7, &total_count, hand_counts);

    std.debug.print("Processed {} total combinations\n", .{total_count});
    std.debug.print("Hash table collisions: {}\n", .{collision_map.count()});
}

// Enumerate all rank combinations for non-flush hands
fn enumerateRankCombinations(mini_table: *[MINI_HASH_SIZE]u16, collision_map: *std.AutoHashMap(u16, u16), counts: *[13]u8, rank: usize, remaining: u8, total_count: *u32, hand_counts: *HandTypeCounts) !void {
    if (rank == 13) {
        if (remaining != 0) return;
        total_count.* += 1;

        // Convert counts to bit representation for key generation
        const bits = countsToCardBits(counts.*);

        // Skip if this would be a flush (handle in flush table)
        if (wouldBeFlush(bits)) return;

        // Generate compressed key
        const key = extractMiniRankKey(bits);
        if (key >= MINI_HASH_SIZE) return;

        // Evaluate hand strength
        const strength = evaluateRankCombination(counts.*, hand_counts);

        // Handle collisions
        if (mini_table[key] == 0) {
            mini_table[key] = strength;
        } else if (mini_table[key] != strength) {
            // Collision detected - use higher strength
            const higher = @max(mini_table[key], strength);
            mini_table[key] = higher;
            try collision_map.put(key, higher);
        }

        return;
    }

    const max = if (remaining > 4) 4 else remaining;
    for (0..max + 1) |c| {
        counts[rank] = @intCast(c);
        try enumerateRankCombinations(mini_table, collision_map, counts, rank + 1, remaining - @as(u8, @intCast(c)), total_count, hand_counts);
    }
}

// Convert rank counts to card bits representation
fn countsToCardBits(counts: [13]u8) u64 {
    var bits: u64 = 0;
    for (counts, 0..) |count, rank| {
        // Set bits for each card of this rank
        const rank_base = rank * 4;
        for (0..count) |suit| {
            bits |= @as(u64, 1) << @intCast(rank_base + suit);
        }
    }
    return bits;
}

// Check if combination would result in a flush
fn wouldBeFlush(bits: u64) bool {
    const suit_bits = bits & FLUSH_CHECK_MASK;
    const suit_counts = suit_bits *% 0x1111_1111_1111_1111;
    return (suit_counts & 0x8888_8888_8888_8888) != 0;
}

// Compress flush rank mask to 10-bit key
fn compressFlushKey(rank_mask: u16) u16 {
    // Extract top 5 ranks for flush evaluation
    var result: u16 = 0;
    var remaining = rank_mask;
    var pos: u4 = 0;

    // Pack 5 highest ranks into 10 bits (2 bits each for rank encoding)
    while (pos < 5 and remaining != 0) : (pos += 1) {
        const highest = 15 - @clz(remaining);
        result |= (@as(u16, @intCast(highest)) & 0x3) << (pos * 2);
        remaining &= ~(@as(u16, 1) << @intCast(highest));
    }

    return result;
}

// Extract mini rank key (11 bits) - matching ultra-fast evaluator logic
fn extractMiniRankKey(bits: u64) u16 {
    // Extract rank presence (13 bits)
    const ranks = bits | (bits >> 13) | (bits >> 26) | (bits >> 39);
    const rank_mask = ranks & 0x1FFF;

    // Fast pair detection
    const pairs = bits & (bits >> 1) & (bits >> 2) & (bits >> 3);
    const has_quad = pairs & RANK_EXTRACT_MASK;
    const has_pair = (bits & (bits >> 1)) & RANK_EXTRACT_MASK;

    // Compress: 8 bits rank + 3 bits pattern type
    const compressed_ranks = compressRanks(rank_mask);
    const pattern_type = if (has_quad != 0) 7 else if (@popCount(has_pair) >= 2) 6 else @as(u3, @intCast(@popCount(has_pair)));

    return (@as(u16, compressed_ranks) << 3) | pattern_type;
}

// Compress 13-bit rank mask to 8 bits
fn compressRanks(rank_mask: u64) u8 {
    const high_cards = (rank_mask >> 8) & 0x1F; // bits 12-8
    const low_cards = rank_mask & 0xFF; // bits 7-0
    return @as(u8, @intCast((high_cards << 3) | (low_cards >> 5)));
}

// Evaluate flush hand strength
fn evaluateFlush(rank_mask: u16, hand_counts: *HandTypeCounts) u16 {
    // Check for straight flush
    if (isFlushStraight(rank_mask)) {
        const straight_high = getFlushStraightHigh(rank_mask);
        if (straight_high == 12) { // A-K-Q-J-10
            hand_counts.straight_flush_count += 1;
            return @intFromEnum(HandStrength.royal_flush);
        } else {
            hand_counts.straight_flush_count += 1;
            return @intFromEnum(HandStrength.straight_flush) + straight_high;
        }
    }

    // Regular flush - get 5 highest cards
    hand_counts.flush_count += 1;
    var value = @intFromEnum(HandStrength.flush);
    var remaining = rank_mask;
    var cards_taken: u8 = 0;

    while (cards_taken < 5 and remaining != 0) : (cards_taken += 1) {
        const highest = 15 - @clz(remaining);
        value += @as(u16, @intCast(highest)) << @as(u4, @intCast(4 - cards_taken));
        remaining &= ~(@as(u16, 1) << @intCast(highest));
    }

    return value;
}

// Check if flush has straight
fn isFlushStraight(rank_mask: u16) bool {
    // Check for 5 consecutive bits
    var r: i8 = 12;
    while (r >= 4) {
        const mask = @as(u16, 0x1F) << @intCast(r - 4);
        if ((rank_mask & mask) == mask) return true;
        r -= 1;
    }

    // Check for wheel (A-2-3-4-5)
    return (rank_mask & 0x100F) == 0x100F;
}

fn getFlushStraightHigh(rank_mask: u16) u8 {
    // Find highest straight
    var r: i8 = 12;
    while (r >= 4) {
        const mask = @as(u16, 0x1F) << @intCast(r - 4);
        if ((rank_mask & mask) == mask) return @intCast(r);
        r -= 1;
    }

    // Wheel
    if ((rank_mask & 0x100F) == 0x100F) return 3;
    return 0;
}

// Evaluate rank combination
fn evaluateRankCombination(counts: [13]u8, hand_counts: *HandTypeCounts) u16 {
    // Count rank frequencies
    var quads: i8 = -1;
    var trips: [2]i8 = [_]i8{-1} ** 2;
    var trip_count: usize = 0;
    var pairs: [3]i8 = [_]i8{-1} ** 3;
    var pair_count: usize = 0;
    var singles: [7]i8 = [_]i8{-1} ** 7;
    var single_count: usize = 0;

    // Scan from high to low (Ace to 2)
    var r: i8 = 12;
    while (r >= 0) : (r -= 1) {
        switch (counts[@intCast(r)]) {
            4 => quads = r,
            3 => {
                if (trip_count < 2) {
                    trips[trip_count] = r;
                    trip_count += 1;
                }
            },
            2 => {
                if (pair_count < 3) {
                    pairs[pair_count] = r;
                    pair_count += 1;
                }
            },
            1 => {
                if (single_count < 7) {
                    singles[single_count] = r;
                    single_count += 1;
                }
            },
            else => {},
        }
    }

    // Check for straights
    const is_straight = checkStraightFromCounts(counts);

    // Classify hand type and return base value
    if (quads >= 0) {
        hand_counts.quad_count += 1;
        return @intFromEnum(HandStrength.four_kind) + @as(u16, @intCast(quads));
    } else if (trips[0] >= 0 and (trips[1] >= 0 or pairs[0] >= 0)) {
        hand_counts.full_count += 1;
        return @intFromEnum(HandStrength.full_house) + @as(u16, @intCast(trips[0]));
    } else if (is_straight) {
        hand_counts.straight_count += 1;
        return @intFromEnum(HandStrength.straight) + getStraightRank(counts);
    } else if (trips[0] >= 0) {
        hand_counts.trips_count += 1;
        return @intFromEnum(HandStrength.three_kind) + @as(u16, @intCast(trips[0]));
    } else if (pair_count >= 2) {
        hand_counts.two_pair_count += 1;
        return @intFromEnum(HandStrength.two_pair) + @as(u16, @intCast(pairs[0]));
    } else if (pairs[0] >= 0) {
        hand_counts.one_pair_count += 1;
        return @intFromEnum(HandStrength.pair) + @as(u16, @intCast(pairs[0]));
    } else {
        hand_counts.high_card_count += 1;
        return @intFromEnum(HandStrength.high_card) + @as(u16, @intCast(singles[0]));
    }
}

fn checkStraightFromCounts(counts: [13]u8) bool {
    // Build rank mask
    var rank_mask: u32 = 0;
    for (counts, 0..) |count, rank| {
        if (count > 0) rank_mask |= @as(u32, 1) << @intCast(rank);
    }

    // Check for 5 consecutive ranks
    var r: i8 = 12;
    while (r >= 4) {
        const mask = @as(u32, 0x1F) << @intCast(r - 4);
        if ((rank_mask & mask) == mask) return true;
        r -= 1;
    }

    // Check for wheel (A-2-3-4-5)
    return (rank_mask & 0x100F) == 0x100F;
}

fn getStraightRank(counts: [13]u8) u16 {
    var rank_mask: u32 = 0;
    for (counts, 0..) |count, rank| {
        if (count > 0) rank_mask |= @as(u32, 1) << @intCast(rank);
    }

    // Find highest straight
    var r: i8 = 12;
    while (r >= 4) {
        const mask = @as(u32, 0x1F) << @intCast(r - 4);
        if ((rank_mask & mask) == mask) return @as(u16, @intCast(r));
        r -= 1;
    }

    // Wheel
    if ((rank_mask & 0x100F) == 0x100F) return 3;
    return 0;
}

// Write tables to source file
fn writeTables(mini_table: *const [MINI_HASH_SIZE]u16, flush_table: *const [FLUSH_TABLE_SIZE]u16) !void {
    const file = try std.fs.cwd().createFile("src/tables.zig", .{});
    defer file.close();
    const w = file.writer();

    try w.writeAll("// Auto-generated lookup tables for L1 cache optimization\n");
    try w.writeAll("// DO NOT EDIT - Generated by build_tables.zig\n");
    try w.writeAll("// Total size: ~6KB - fits comfortably in L1 cache\n\n");

    // Write mini hash table
    try w.print("pub const MINI_HASH_TABLE = [_]u16{{\n", .{});
    for (mini_table, 0..) |value, i| {
        if (i % 16 == 0) try w.writeAll("    ");
        try w.print("{},", .{value});
        if (i % 16 == 15) try w.writeAll("\n");
    }
    try w.writeAll("\n};\n\n");

    // Write flush table
    try w.print("pub const FLUSH_TABLE = [_]u16{{\n", .{});
    for (flush_table, 0..) |value, i| {
        if (i % 16 == 0) try w.writeAll("    ");
        try w.print("{},", .{value});
        if (i % 16 == 15) try w.writeAll("\n");
    }
    try w.writeAll("\n};\n\n");

    // Write constants
    try w.writeAll("// Table sizes for validation\n");
    try w.print("pub const MINI_HASH_SIZE = {};\n", .{MINI_HASH_SIZE});
    try w.print("pub const FLUSH_TABLE_SIZE = {};\n", .{FLUSH_TABLE_SIZE});
    try w.print("pub const TOTAL_SIZE_KB = {};\n", .{(MINI_HASH_SIZE + FLUSH_TABLE_SIZE) * 2 / 1024});
}

fn printHandCounts(hand_counts: *const HandTypeCounts) void {
    std.debug.print("\nHand type distribution:\n", .{});
    std.debug.print("  High card: {}\n", .{hand_counts.high_card_count});
    std.debug.print("  One pair: {}\n", .{hand_counts.one_pair_count});
    std.debug.print("  Two pair: {}\n", .{hand_counts.two_pair_count});
    std.debug.print("  Three of a kind: {}\n", .{hand_counts.trips_count});
    std.debug.print("  Straight: {}\n", .{hand_counts.straight_count});
    std.debug.print("  Flush: {}\n", .{hand_counts.flush_count});
    std.debug.print("  Full house: {}\n", .{hand_counts.full_count});
    std.debug.print("  Four of a kind: {}\n", .{hand_counts.quad_count});
    std.debug.print("  Straight flush: {}\n", .{hand_counts.straight_flush_count});
}
