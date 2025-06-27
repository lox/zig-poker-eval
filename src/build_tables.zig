const std = @import("std");
const print = std.debug.print;
const evaluator = @import("slow_evaluator");

// Table generation constants
const CHD_NUM_BUCKETS = 8192; // 2^13 buckets
const CHD_TABLE_SIZE = 131072; // 2^17 slots
const CHD_EXPECTED_PATTERNS = 49205; // Non-flush patterns
const FLUSH_PATTERNS = 1287; // C(13,5)

// Generated table data
var chd_g_array: [CHD_NUM_BUCKETS]u8 = undefined;
var chd_value_table: [CHD_TABLE_SIZE]u16 = undefined;
var flush_lookup_table: [8192]u16 = undefined;
var chd_magic_constant: u64 = 0x9e3779b97f4a7c15;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Building poker evaluation tables...\n", .{});

    // Build both tables
    try buildTables(allocator);

    // Generate output file
    print("Writing src/tables.zig...\n", .{});
    try writeTablesFile();

    print("Table generation complete!\n", .{});
}

// Unit tests
test "CHD hash function" {
    const test_rpc: u32 = 0x12345;
    const hash_result = chdHash(test_rpc);
    
    try std.testing.expect(hash_result.bucket < CHD_NUM_BUCKETS);
    try std.testing.expect(hash_result.base_index < CHD_TABLE_SIZE);
}

test "RPC computation" {
    // Test a simple hand: AsKsQsJsTs (royal flush in spades)
    const hand: evaluator.Hand = 
        evaluator.makeCard(0, 12) | // As
        evaluator.makeCard(0, 11) | // Ks  
        evaluator.makeCard(0, 10) | // Qs
        evaluator.makeCard(0, 9) |  // Js
        evaluator.makeCard(0, 8) |  // Ts
        evaluator.makeCard(1, 7) |  // 8h
        evaluator.makeCard(2, 6);   // 7d
    
    const rpc = computeRPC(hand);
    
    // This should be a valid RPC value
    try std.testing.expect(rpc > 0);
    
    // Test that ranks have proper counts (5 spades + 2 other suits)
    // Should have rank counts: A=1, K=1, Q=1, J=1, T=1, 8=1, 7=1, others=0
    // In base-5: this should produce a specific pattern
}

fn buildTables(allocator: std.mem.Allocator) !void {
    // Enumerate all 7-card hands once
    var non_flush_patterns = std.ArrayList(Pattern).init(allocator);
    defer non_flush_patterns.deinit();

    var flush_patterns = std.AutoHashMap(u16, u16).init(allocator);
    defer flush_patterns.deinit();

    var seen_rpcs = std.AutoHashMap(u32, u16).init(allocator);
    defer seen_rpcs.deinit();

    print("Enumerating all 7-card hands...\n", .{});

    var iterator = HandIterator.init();
    var total_hands: usize = 0;

    while (iterator.next()) |hand| {
        total_hands += 1;
        if (total_hands % 10_000_000 == 0) {
            print("  Processed {} million hands...\n", .{total_hands / 1_000_000});
        }

        if (evaluator.hasFlush(hand)) {
            // Handle flush pattern
            const suits = evaluator.getSuitMasks(hand);
            for (suits) |suit| {
                if (@popCount(suit) >= 5) {
                    const pattern = getTop5Ranks(suit);
                    if (!flush_patterns.contains(pattern)) {
                        const rank = evaluateFlushPattern(pattern);
                        try flush_patterns.put(pattern, rank);
                    }
                    break;
                }
            }
        } else {
            // Handle non-flush pattern
            const rpc = computeRPC(hand);
            if (!seen_rpcs.contains(rpc)) {
                const rank = evaluator.evaluateHand(hand);
                try seen_rpcs.put(rpc, rank);
                try non_flush_patterns.append(.{ .key = rpc, .rank = rank });
            }
        }
    }

    print("  Found {} non-flush and {} flush patterns\n", .{
        non_flush_patterns.items.len,
        flush_patterns.count(),
    });

    // Verify critical invariants
    std.debug.assert(non_flush_patterns.items.len == CHD_EXPECTED_PATTERNS);
    std.debug.assert(flush_patterns.count() == FLUSH_PATTERNS);
    print("✓ Pattern counts verified\n", .{});

    // Build CHD for non-flush
    print("Building CHD table...\n", .{});
    try buildCHD(allocator, non_flush_patterns.items);

    // Build direct lookup for flush
    print("Building flush lookup table...\n", .{});
    flush_lookup_table = std.mem.zeroes(@TypeOf(flush_lookup_table));
    var iter = flush_patterns.iterator();
    while (iter.next()) |entry| {
        flush_lookup_table[entry.key_ptr.*] = entry.value_ptr.*;
    }
}

fn buildCHD(allocator: std.mem.Allocator, patterns: []const Pattern) !void {
    // Try different seeds until one works
    for (0..10) |attempt| {
        chd_magic_constant = 0x9e3779b97f4a7c15 +% (attempt * 0x123456789abcdef);

        // Reset tables
        chd_g_array = std.mem.zeroes(@TypeOf(chd_g_array));
        chd_value_table = std.mem.zeroes(@TypeOf(chd_value_table));

        // Group patterns by bucket
        var buckets: [CHD_NUM_BUCKETS]std.ArrayList(Pattern) = undefined;
        for (&buckets) |*bucket| {
            bucket.* = std.ArrayList(Pattern).init(allocator);
        }
        defer {
            for (&buckets) |*bucket| bucket.deinit();
        }

        for (patterns) |pattern| {
            const h = chdHash(pattern.key);
            try buckets[h.bucket].append(pattern);
        }

        // Try to find displacement for each bucket
        var occupied = [_]bool{false} ** CHD_TABLE_SIZE;
        var success = true;

        // Process buckets in order of decreasing size
        var bucket_order: [CHD_NUM_BUCKETS]u16 = undefined;
        for (0..CHD_NUM_BUCKETS) |i| {
            bucket_order[i] = @intCast(i);
        }
        std.sort.pdq(u16, &bucket_order, buckets, bucketSizeDesc);

        for (bucket_order) |bucket_id| {
            const bucket = &buckets[bucket_id];
            if (bucket.items.len == 0) continue;

            const displacement = findDisplacement(bucket.items, &occupied) orelse {
                success = false;
                break;
            };

            chd_g_array[bucket_id] = @intCast(displacement);

            // Place all entries
            for (bucket.items) |pattern| {
                const h = chdHash(pattern.key);
                const slot = (h.base_index + displacement) & (CHD_TABLE_SIZE - 1);
                occupied[slot] = true;
                chd_value_table[slot] = pattern.rank;
            }
        }

        if (success) {
            print("  CHD built successfully with seed attempt {}\n", .{attempt + 1});
            return;
        }
    }

    return error.CHDConstructionFailed;
}

fn findDisplacement(patterns: []const Pattern, occupied: *[CHD_TABLE_SIZE]bool) ?u32 {
    for (0..256) |d| {
        var collision = false;
        for (patterns) |pattern| {
            const h = chdHash(pattern.key);
            const slot = (h.base_index + d) & (CHD_TABLE_SIZE - 1);
            if (occupied[slot]) {
                collision = true;
                break;
            }
        }
        if (!collision) return @intCast(d);
    }
    return null;
}

// Helper functions
fn computeRPC(hand: evaluator.Hand) u32 {
    const suits = evaluator.getSuitMasks(hand);
    var rank_counts = [_]u8{0} ** 13;

    for (0..13) |rank| {
        const rank_bit = @as(u16, 1) << @intCast(rank);
        for (suits) |suit| {
            if ((suit & rank_bit) != 0) rank_counts[rank] += 1;
        }
    }

    // Base-5 encoding
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }
    return rpc;
}

fn chdHash(key: u32) struct { bucket: u32, base_index: u32 } {
    var h = @as(u64, key);
    h ^= h >> 33;
    h *%= chd_magic_constant;
    h ^= h >> 29;

    return .{
        .bucket = @intCast(h >> 51),
        .base_index = @intCast(h & 0x1FFFF),
    };
}

fn getTop5Ranks(suit_mask: u16) u16 {
    if (@popCount(suit_mask) == 5) return suit_mask;

    // Check for straights first
    const straights = [_]u16{
        0x1F00, 0x0F80, 0x07C0, 0x03E0, 0x01F0,
        0x00F8, 0x007C, 0x003E, 0x001F, 0x100F,
    };

    for (straights) |pattern| {
        if ((suit_mask & pattern) == pattern) return pattern;
    }

    // Take highest 5 ranks
    var result: u16 = 0;
    var count: u8 = 0;
    var rank: i8 = 12;

    while (count < 5 and rank >= 0) : (rank -= 1) {
        const bit = @as(u16, 1) << @intCast(rank);
        if ((suit_mask & bit) != 0) {
            result |= bit;
            count += 1;
        }
    }

    return result;
}

fn evaluateFlushPattern(pattern: u16) u16 {
    // Build minimal hand with this flush pattern
    var hand: evaluator.Hand = 0;

    // Add flush cards in clubs
    for (0..13) |r| {
        if ((pattern & (@as(u16, 1) << @intCast(r))) != 0) {
            hand |= evaluator.makeCard(0, @intCast(r));
        }
    }

    // Add 2 non-conflicting cards from other suits
    var added: u8 = 0;
    for (0..13) |r| {
        if ((pattern & (@as(u16, 1) << @intCast(r))) == 0 and added < 2) {
            hand |= evaluator.makeCard(1 + added, @intCast(r));
            added += 1;
        }
    }

    return evaluator.evaluateHand(hand);
}

fn writeTablesFile() !void {
    const file = try std.fs.cwd().createFile("src/tables.zig", .{});
    defer file.close();
    const w = file.writer();

    try w.print("// Generated lookup tables for poker evaluator\n\n", .{});

    // Write CHD tables
    try w.print("pub const chd_g_array = [_]u8{{\n", .{});
    for (chd_g_array, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    try w.print("pub const chd_value_table = [_]u16{{\n", .{});
    for (chd_value_table, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    try w.print("pub const flush_lookup_table = [_]u16{{\n", .{});
    for (flush_lookup_table, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    // Write all CHD constants
    try w.print("// CHD constants\n", .{});
    try w.print("pub const CHD_MAGIC_CONSTANT: u64 = 0x{X};\n", .{chd_magic_constant});
    try w.print("pub const CHD_NUM_BUCKETS: u32 = {};\n", .{CHD_NUM_BUCKETS});
    try w.print("pub const CHD_TABLE_SIZE: u32 = {};\n", .{CHD_TABLE_SIZE});
    
    // Add compile-time size validation
    try w.print("\n// Compile-time size validation\n", .{});
    try w.print("comptime {{\n", .{});
    try w.print("    const std = @import(\"std\");\n", .{});
    try w.print("    std.debug.assert(@sizeOf(@TypeOf(chd_g_array)) == {});\n", .{CHD_NUM_BUCKETS});
    try w.print("    std.debug.assert(@sizeOf(@TypeOf(chd_value_table)) == {} * @sizeOf(u16));\n", .{CHD_TABLE_SIZE});
    try w.print("    std.debug.assert(@sizeOf(@TypeOf(flush_lookup_table)) == 8192 * @sizeOf(u16));\n", .{});
    try w.print("}}\n", .{});
}

// Simple types
const Pattern = struct {
    key: u32,
    rank: u16,
};

const HandIterator = struct {
    combination: [7]u8,
    finished: bool,

    fn init() HandIterator {
        return .{
            .combination = [_]u8{ 0, 1, 2, 3, 4, 5, 6 },
            .finished = false,
        };
    }

    fn next(self: *HandIterator) ?evaluator.Hand {
        if (self.finished) return null;

        var hand: evaluator.Hand = 0;
        for (self.combination) |card_idx| {
            hand |= evaluator.makeCard(card_idx / 13, card_idx % 13);
        }

        if (!self.nextCombination()) self.finished = true;
        return hand;
    }

    fn nextCombination(self: *HandIterator) bool {
        var i: i8 = 6;
        while (i >= 0) : (i -= 1) {
            const idx = @as(usize, @intCast(i));
            if (self.combination[idx] < 52 - (7 - idx)) {
                self.combination[idx] += 1;
                for (idx + 1..7) |j| {
                    self.combination[j] = self.combination[j - 1] + 1;
                }
                return true;
            }
        }
        return false;
    }
};

fn bucketSizeDesc(buckets: [CHD_NUM_BUCKETS]std.ArrayList(Pattern), a: u16, b: u16) bool {
    return buckets[a].items.len > buckets[b].items.len;
}
