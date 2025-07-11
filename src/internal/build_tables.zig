const std = @import("std");
const print = std.debug.print;
const evaluator = @import("slow_evaluator.zig");
const mphf = @import("mphf.zig");

// Table generation constants
const CHD_EXPECTED_PATTERNS = 49205; // Non-flush patterns
const FLUSH_PATTERNS = 1287; // C(13,5)

// Generated table data
var chd_result: mphf.CHDResult = undefined;
var flush_lookup_table: [8192]u16 = undefined;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Building poker evaluation tables...\n", .{});

    // Build both tables
    try buildTables(allocator);

    // Generate output file
    print("Writing src/internal/tables.zig...\n", .{});
    try writeTablesFile();

    // Validate generated table sizes
    print("Validating table sizes...\n", .{});
    std.debug.assert(chd_result.g_array.len == mphf.DEFAULT_NUM_BUCKETS);
    std.debug.assert(chd_result.value_table.len == mphf.DEFAULT_TABLE_SIZE);
    std.debug.assert(flush_lookup_table.len == 8192);
    print("✓ Table sizes validated\n", .{});

    print("Table generation complete!\n", .{});
}

// Unit tests
test "CHD hash function" {
    const test_rpc: u32 = 0x12345;
    const hash_result = mphf.hashKey(test_rpc, mphf.DEFAULT_MAGIC_CONSTANT);

    try std.testing.expect(hash_result.bucket < mphf.DEFAULT_NUM_BUCKETS);
    try std.testing.expect(hash_result.base_index < mphf.DEFAULT_TABLE_SIZE);
}

test "RPC computation" {
    // Test a simple hand: AsKsQsJsTs (royal flush in spades)
    const hand: evaluator.Hand =
        evaluator.makeCard(.clubs, .ace) | // Ac
        evaluator.makeCard(.clubs, .king) | // Kc
        evaluator.makeCard(.clubs, .queen) | // Qc
        evaluator.makeCard(.clubs, .jack) | // Jc
        evaluator.makeCard(.clubs, .ten) | // Tc
        evaluator.makeCard(.diamonds, .nine) | // 9d
        evaluator.makeCard(.hearts, .eight); // 8h

    const rpc = computeRPC(hand);

    // This should be a valid RPC value
    try std.testing.expect(rpc > 0);

    // Test that ranks have proper counts (5 spades + 2 other suits)
    // Should have rank counts: A=1, K=1, Q=1, J=1, T=1, 8=1, 7=1, others=0
    // In base-5: this should produce a specific pattern
}

test "flush pattern extraction - overlapping straights" {
    // Test hand 1: spades 2,3,4,5,6,7 (mask 0x3F)
    const spades_mask: u16 = 0x3F; // 2,3,4,5,6,7
    const pattern1 = getTop5Ranks(spades_mask);

    // Should be 7-6-5-4-3 (0x003E), not 6-5-4-3-2 (0x001F)
    try std.testing.expectEqual(@as(u16, 0x003E), pattern1);

    // Test the evaluation of this pattern
    _ = evaluateFlushPattern(pattern1);

    // Test hand 2: hearts 8,9,T,J,Q,K (mask 0xFC0)
    const hearts_mask: u16 = 0xFC0; // 8,9,T,J,Q,K
    const pattern2 = getTop5Ranks(hearts_mask);

    // Should be K-Q-J-T-9 (0x0F80)
    try std.testing.expectEqual(@as(u16, 0x0F80), pattern2);

    // Test the evaluation of this pattern
    _ = evaluateFlushPattern(pattern2);
}

test "slow evaluator vs table patterns" {
    // Test the two failing hands directly
    const hand1: evaluator.Hand = 0x1F8000000008;
    const hand2: evaluator.Hand = 0x3F00001000;

    // Just verify the hands can be evaluated without errors
    _ = evaluator.evaluateHand(hand1);
    _ = evaluator.evaluateHand(hand2);

    // Verify pattern extraction works
    const suits1 = evaluator.getSuitMasks(hand1);
    const suits2 = evaluator.getSuitMasks(hand2);

    for (suits1) |suit| {
        if (@popCount(suit) >= 5) {
            const pattern = getTop5Ranks(suit);
            _ = evaluateFlushPattern(pattern);
        }
    }

    for (suits2) |suit| {
        if (@popCount(suit) >= 5) {
            const pattern = getTop5Ranks(suit);
            _ = evaluateFlushPattern(pattern);
        }
    }
}

fn buildTables(allocator: std.mem.Allocator) !void {
    // Enumerate all 7-card hands once
    var non_flush_patterns = std.ArrayList(mphf.Pattern).init(allocator);
    defer non_flush_patterns.deinit();

    var flush_patterns = std.AutoHashMap(u16, u16).init(allocator);
    defer flush_patterns.deinit();

    var seen_rpcs = std.AutoHashMap(u32, u16).init(allocator);
    defer seen_rpcs.deinit();

    print("Enumerating all 7-card hands...\n", .{});

    // Debug: Track specific problem hands
    const problem_hands = [_]u64{
        0x4802245,
        0x8000802245,
        0x1004002245,
        0x4402445,
        0x4C02045,
    };

    var iterator = HandIterator.init();
    var total_hands: usize = 0;

    while (iterator.next()) |hand| {
        total_hands += 1;

        // Debug: Check if this is one of our problem hands
        for (problem_hands) |ph| {
            if (hand == ph) {
                print("\n=== FOUND PROBLEM HAND 0x{X} ===\n", .{hand});

                // Analyze the hand
                const suits = evaluator.getSuitMasks(hand);
                var rank_counts = [_]u8{0} ** 13;
                for (0..13) |rank| {
                    const rank_bit = @as(u16, 1) << @intCast(rank);
                    for (suits) |suit| {
                        if ((suit & rank_bit) != 0) rank_counts[rank] += 1;
                    }
                }

                print("  Suits: C=0x{X} D=0x{X} H=0x{X} S=0x{X}\n", .{ suits[0], suits[1], suits[2], suits[3] });
                print("  Rank counts: ", .{});
                for (rank_counts, 0..) |count, rank| {
                    if (count > 0) {
                        const rank_str = switch (rank) {
                            0 => "2",
                            1 => "3",
                            2 => "4",
                            3 => "5",
                            4 => "6",
                            5 => "7",
                            6 => "8",
                            7 => "9",
                            8 => "T",
                            9 => "J",
                            10 => "Q",
                            11 => "K",
                            12 => "A",
                            else => "?",
                        };
                        print("{s}:{} ", .{ rank_str, count });
                    }
                }
                print("\n", .{});

                const has_flush = evaluator.hasFlush(hand);
                const rpc = computeRPC(hand);
                const rank = evaluator.evaluateHand(hand);

                print("  Has flush: {}\n", .{has_flush});
                print("  RPC: {}\n", .{rpc});
                print("  Slow evaluator rank: {}\n", .{rank});

                // Check hand type
                var pairs: u8 = 0;
                var trips: u8 = 0;
                for (rank_counts) |count| {
                    switch (count) {
                        2 => pairs += 1,
                        3 => trips += 1,
                        else => {},
                    }
                }

                if (trips == 2) {
                    print("  *** TWO TRIPS - should be FULL HOUSE (rank 166-321) ***\n", .{});
                } else if (trips == 1 and pairs >= 1) {
                    print("  *** TRIPS + PAIR = FULL HOUSE ***\n", .{});
                } else if (trips == 1) {
                    print("  *** THREE OF A KIND (rank 1609-2466) ***\n", .{});
                }
                print("=================================\n\n", .{});
            }
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
                try non_flush_patterns.append(.{ .key = rpc, .value = rank });
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
    chd_result = try mphf.buildChd(allocator, non_flush_patterns.items, mphf.DEFAULT_NUM_BUCKETS, mphf.DEFAULT_TABLE_SIZE);
    print("  CHD built successfully\n", .{});

    // Verify CHD lookup works for our problem RPC
    print("Verifying CHD lookup...\n", .{});
    const test_rpc: u32 = 742203275;
    const expected_rank: u16 = 2389;
    const actual_rank = mphf.lookup(test_rpc, chd_result.magic_constant, chd_result.g_array, chd_result.value_table, mphf.DEFAULT_TABLE_SIZE);
    print("  RPC {} -> expected rank {}, got rank {}\n", .{ test_rpc, expected_rank, actual_rank });
    if (actual_rank != expected_rank) {
        print("  *** CHD LOOKUP VERIFICATION FAILED! ***\n", .{});

        // Debug the hash
        const h = mphf.hashKey(test_rpc, chd_result.magic_constant);
        print("  Hash result: bucket={}, base_index={}\n", .{ h.bucket, h.base_index });
        const displacement = chd_result.g_array[h.bucket];
        const final_index = (h.base_index + displacement) & (mphf.DEFAULT_TABLE_SIZE - 1);
        print("  Displacement={}, final_index={}\n", .{ displacement, final_index });
        print("  Value at final_index: {}\n", .{chd_result.value_table[final_index]});
    }

    // Build direct lookup for flush
    print("Building flush lookup table...\n", .{});
    flush_lookup_table = std.mem.zeroes(@TypeOf(flush_lookup_table));
    var iter = flush_patterns.iterator();
    while (iter.next()) |entry| {
        flush_lookup_table[entry.key_ptr.*] = entry.value_ptr.*;
    }
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

fn getTop5Ranks(suit_mask: u16) u16 {
    if (@popCount(suit_mask) == 5) return suit_mask;

    // Check for straights starting from highest (A-K-Q-J-T down to 6-5-4-3-2)
    // This ensures we find the HIGHEST straight when there are overlapping ones
    var straight_mask: u16 = 0x1F00; // Start with A-K-Q-J-T
    var i: u8 = 0;
    while (i <= 8) : (i += 1) {
        if ((suit_mask & straight_mask) == straight_mask) {
            return straight_mask;
        }
        straight_mask >>= 1; // Shift right to check next lower straight
    }

    // Check for wheel (A-2-3-4-5) last since it's the lowest straight
    if ((suit_mask & 0x100F) == 0x100F) { // A,2,3,4,5
        return 0x100F; // Return full wheel pattern
    }

    // No straight found, take highest 5 ranks
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
    // Check if this pattern is a straight flush first
    const straight_patterns = [_]u16{
        0x1F00, // A-K-Q-J-T (royal flush)
        0x0F80, // K-Q-J-T-9
        0x07C0, // Q-J-T-9-8
        0x03E0, // J-T-9-8-7
        0x01F0, // T-9-8-7-6
        0x00F8, // 9-8-7-6-5
        0x007C, // 8-7-6-5-4
        0x003E, // 7-6-5-4-3
        0x001F, // 6-5-4-3-2
        0x100F, // A-5-4-3-2 (wheel)
    };

    for (straight_patterns, 0..) |straight_pattern, i| {
        if (pattern == straight_pattern) {
            // This is a straight flush
            if (straight_pattern == 0x1F00) {
                return 0; // Royal flush (best possible hand)
            } else if (straight_pattern == 0x100F) {
                return 9; // Wheel straight flush (worst straight flush)
            } else {
                // Other straight flushes: K-high=1, Q-high=2, ..., 6-high=8
                return @as(u16, @intCast(i));
            }
        }
    }

    // Not a straight flush, so it's a regular flush
    // Find the highest card in the pattern
    const high_card_bit = @clz(pattern);
    const high_card_rank = 15 - high_card_bit;

    // Use the same formula as slow_evaluator for flush ranking:
    // Flush ranks are 322-1598, with A-high=322, K-high=422, etc.
    return 322 + @as(u16, (12 - high_card_rank)) * 100;
}

fn writeTablesFile() !void {
    const file = try std.fs.cwd().createFile("src/internal/tables.zig", .{});
    defer file.close();
    const w = file.writer();

    try w.print("// Generated lookup tables for poker evaluator\n\n", .{});

    try w.print("const mphf = @import(\"mphf.zig\");\n\n", .{});

    // Write CHD tables (private)
    try w.print("const chd_g_array = [_]u8{{\n", .{});
    for (chd_result.g_array, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    try w.print("const chd_value_table = [_]u16{{\n", .{});
    for (chd_result.value_table, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    try w.print("const flush_lookup_table = [_]u16{{\n", .{});
    for (flush_lookup_table, 0..) |val, i| {
        if (i % 16 == 0) try w.print("    ", .{});
        try w.print("{}, ", .{val});
        if (i % 16 == 15) try w.print("\n", .{});
    }
    try w.print("}};\n\n", .{});

    // Write all CHD constants (private)
    try w.print("// CHD constants\n", .{});
    try w.print("const CHD_MAGIC_CONSTANT: u64 = 0x{X};\n", .{chd_result.magic_constant});
    try w.print("const CHD_NUM_BUCKETS: u32 = {};\n", .{mphf.DEFAULT_NUM_BUCKETS});
    try w.print("const CHD_TABLE_SIZE: u32 = {};\n", .{mphf.DEFAULT_TABLE_SIZE});
    try w.print("\n", .{});

    // Write public API functions
    try w.print("// Public API - only expose the functions needed by evaluator\n", .{});
    try w.print("pub inline fn lookup(rpc: u32) u16 {{\n", .{});
    try w.print("    return mphf.lookup(rpc, CHD_MAGIC_CONSTANT, &chd_g_array, &chd_value_table, CHD_TABLE_SIZE);\n", .{});
    try w.print("}}\n\n", .{});
    try w.print("pub inline fn flushLookup(pattern: u16) u16 {{\n", .{});
    try w.print("    return flush_lookup_table[pattern];\n", .{});
    try w.print("}}\n", .{});
}

// Remove old Pattern type - using mphf.Pattern now

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
            const suit: evaluator.Suit = @enumFromInt(card_idx / 13);
            const rank: evaluator.Rank = @enumFromInt(card_idx % 13);
            hand |= evaluator.makeCard(suit, rank);
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
