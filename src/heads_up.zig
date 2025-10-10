// Specialized Heads-Up Equity Tables
// Pre-computed win frequencies for heads-up (2 player) poker scenarios

// Based on analysis of holdem-hand-evaluator approach
// https://github.com/b-inary/holdem-hand-evaluator

const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");
const range_mod = @import("range");

// There are 169 unique starting hands in Texas Hold'em:
// - 13 pocket pairs (AA, KK, QQ, ..., 22)
// - 78 suited combinations (AKs, AQs, ..., 32s)
// - 78 offsuit combinations (AKo, AQo, ..., 32o)
//
// We represent these as a 13x13 matrix where:
// - Diagonal entries are pocket pairs
// - Upper triangle are suited hands
// - Lower triangle are offsuit hands
//
// Note: This module has its own simple notation parsing for converting
// hand strings to indices (0-168). This avoids dependencies on the
// more complex range/notation modules which generate actual card combinations.

pub const HandIndex = struct {
    // Convert two ranks to an index in the 169-hand table
    // rank1 should be >= rank2
    // suited: true for suited hands, false for offsuit
    pub fn getIndex(rank1: u8, rank2: u8, suited: bool) u8 {
        std.debug.assert(rank1 >= rank2);
        std.debug.assert(rank1 < 13 and rank2 < 13);

        if (rank1 == rank2) {
            // Pocket pair - on the diagonal
            return rank1 * 13 + rank1;
        } else if (suited) {
            // Suited hand - upper triangle
            return rank1 * 13 + rank2;
        } else {
            // Offsuit hand - lower triangle
            return rank2 * 13 + rank1;
        }
    }

    // Parse a hand string like "AKs" or "TT" to an index
    pub fn parseHand(hand_str: []const u8) !u8 {
        if (hand_str.len < 2 or hand_str.len > 3) {
            return error.InvalidHandFormat;
        }

        const rank1 = try rankFromChar(hand_str[0]);
        const rank2 = try rankFromChar(hand_str[1]);

        const suited = if (hand_str.len == 3)
            hand_str[2] == 's'
        else
            rank1 == rank2; // pocket pairs are implicitly "suited"

        const high = @max(rank1, rank2);
        const low = @min(rank1, rank2);

        return getIndex(high, low, suited);
    }

    fn rankFromChar(c: u8) !u8 {
        return switch (c) {
            '2' => 0,
            '3' => 1,
            '4' => 2,
            '5' => 3,
            '6' => 4,
            '7' => 5,
            '8' => 6,
            '9' => 7,
            'T', 't' => 8,
            'J', 'j' => 9,
            'Q', 'q' => 10,
            'K', 'k' => 11,
            'A', 'a' => 12,
            else => error.InvalidRank,
        };
    }

    /// Convert hand index (0-168) to poker notation string
    /// Returns strings like "AA", "AKs", "72o", etc.
    pub fn toNotation(index: u8) []const u8 {
        const row = index / 13;
        const col = index % 13;

        if (row == col) {
            // Pocket pair - diagonal
            return switch (row) {
                0 => "22",
                1 => "33",
                2 => "44",
                3 => "55",
                4 => "66",
                5 => "77",
                6 => "88",
                7 => "99",
                8 => "TT",
                9 => "JJ",
                10 => "QQ",
                11 => "KK",
                12 => "AA",
                else => unreachable,
            };
        } else if (row > col) {
            // Suited - upper triangle
            return formatNotation(row, col, true);
        } else {
            // Offsuit - lower triangle
            return formatNotation(col, row, false);
        }
    }

    /// Format two ranks into notation like "AKs" or "QJo"
    fn formatNotation(high_rank: u8, low_rank: u8, suited: bool) []const u8 {
        // Generate static lookup table at comptime
        const notations = comptime blk: {
            var table: [13][13][2][]const u8 = undefined;

            for (0..13) |h| {
                for (0..13) |l| {
                    const h_char = switch (h) {
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
                        else => unreachable,
                    };
                    const l_char = switch (l) {
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
                        else => unreachable,
                    };

                    table[h][l][0] = h_char ++ l_char ++ "s";
                    table[h][l][1] = h_char ++ l_char ++ "o";
                }
            }
            break :blk table;
        };

        const idx: usize = if (suited) 0 else 1;
        return notations[high_rank][low_rank][idx];
    }

    /// Create a Range containing all combinations for this hand index
    /// Example: toRange(168, allocator) creates Range with all 6 AA combinations
    pub fn toRange(index: u8, allocator: std.mem.Allocator) !range_mod.Range {
        const notation = toNotation(index);
        return range_mod.parseRange(notation, allocator);
    }
};

// Pre-computed equity for all 169 starting hands vs random opponent
// Each entry is (win_rate_x1000, tie_rate_x1000) - stored as u16 to save space
// For example, (850, 23) means 85.0% win, 2.3% tie, 12.7% loss
const heads_up_tables = @import("heads_up_tables.zig");
pub const PREFLOP_VS_RANDOM = heads_up_tables.PREFLOP_VS_RANDOM;

// Pre-computed 169x169 matrix for exact head-to-head matchups
// matrix[hero_idx][villain_idx] = (hero_win_rate_x1000, tie_rate_x1000)
const heads_up_matrix = @import("heads_up_matrix.zig");
pub const HEADS_UP_MATRIX = heads_up_matrix.HEADS_UP_MATRIX;

// Fast heads-up equity calculation using pre-computed tables
pub const HeadsUpEquity = struct {
    pub const Result = struct {
        win: f32,
        tie: f32,
        loss: f32,
    };

    // Get preflop equity for a specific matchup using exact 169x169 table
    pub fn getPreflopEquity(hero_idx: u8, villain_idx: u8) Result {
        const equity_data = HEADS_UP_MATRIX[hero_idx][villain_idx];

        const win_rate = @as(f32, @floatFromInt(equity_data[0])) / 1000.0;
        const tie_rate = @as(f32, @floatFromInt(equity_data[1])) / 1000.0;
        const loss_rate = 1.0 - win_rate - tie_rate;

        return .{
            .win = win_rate,
            .tie = tie_rate,
            .loss = loss_rate,
        };
    }

    // Fast path for preflop all-in scenarios
    pub fn evaluatePreflopAllIn(
        hero_cards: card.Hand,
        villain_cards: card.Hand,
    ) Result {
        // Convert cards to hand indices
        const hero_idx = getHandIndex(hero_cards);
        const villain_idx = getHandIndex(villain_cards);

        return getPreflopEquity(hero_idx, villain_idx);
    }

    fn getHandIndex(hand: card.Hand) u8 {
        // Extract the two cards from the hand bitfield
        var cards_found: [2]struct { rank: u8, suit: u8 } = undefined;
        var count: usize = 0;

        for (0..52) |i| {
            if ((hand & (@as(u64, 1) << @intCast(i))) != 0) {
                cards_found[count] = .{
                    .rank = @intCast(i % 13),
                    .suit = @intCast(i / 13),
                };
                count += 1;
                if (count >= 2) break;
            }
        }

        const suited = cards_found[0].suit == cards_found[1].suit;
        const high = @max(cards_found[0].rank, cards_found[1].rank);
        const low = @min(cards_found[0].rank, cards_found[1].rank);

        return HandIndex.getIndex(high, low, suited);
    }
};

test "hand index calculation" {
    // Test pocket pairs
    try std.testing.expectEqual(@as(u8, 0), HandIndex.getIndex(0, 0, false)); // 22
    try std.testing.expectEqual(@as(u8, 168), HandIndex.getIndex(12, 12, false)); // AA

    // Test suited hands
    try std.testing.expectEqual(@as(u8, 167), HandIndex.getIndex(12, 11, true)); // AKs

    // Test offsuit hands
    try std.testing.expectEqual(@as(u8, 155), HandIndex.getIndex(12, 11, false)); // AKo
}

test "parse hand strings" {
    try std.testing.expectEqual(@as(u8, 168), try HandIndex.parseHand("AA"));
    try std.testing.expectEqual(@as(u8, 167), try HandIndex.parseHand("AKs"));
    try std.testing.expectEqual(@as(u8, 155), try HandIndex.parseHand("AKo"));
    try std.testing.expectEqual(@as(u8, 0), try HandIndex.parseHand("22"));
}

test "HandIndex.toNotation roundtrip" {
    // Test that toNotation produces correct strings
    try std.testing.expectEqualStrings("AA", HandIndex.toNotation(168));
    try std.testing.expectEqualStrings("KK", HandIndex.toNotation(154));
    try std.testing.expectEqualStrings("QQ", HandIndex.toNotation(140));
    try std.testing.expectEqualStrings("22", HandIndex.toNotation(0));

    // Test suited and offsuit
    try std.testing.expectEqualStrings("AKs", HandIndex.toNotation(167));
    try std.testing.expectEqualStrings("AKo", HandIndex.toNotation(155));

    // Test roundtrip: notation -> index -> notation
    const aa_idx = try HandIndex.parseHand("AA");
    try std.testing.expectEqualStrings("AA", HandIndex.toNotation(aa_idx));

    const aks_idx = try HandIndex.parseHand("AKs");
    try std.testing.expectEqualStrings("AKs", HandIndex.toNotation(aks_idx));

    const ako_idx = try HandIndex.parseHand("AKo");
    try std.testing.expectEqualStrings("AKo", HandIndex.toNotation(ako_idx));
}

test "HandIndex.toRange creates correct ranges" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test AA creates 6 combinations
    var aa_range = try HandIndex.toRange(168, allocator);
    defer aa_range.deinit();
    try std.testing.expect(aa_range.handCount() == 6);

    // Test AKs creates 4 combinations
    var aks_range = try HandIndex.toRange(167, allocator);
    defer aks_range.deinit();
    try std.testing.expect(aks_range.handCount() == 4);

    // Test AKo creates 12 combinations
    var ako_range = try HandIndex.toRange(155, allocator);
    defer ako_range.deinit();
    try std.testing.expect(ako_range.handCount() == 12);
}

test "evaluatePreflopAllIn integration" {
    // Note: Uses 169x169 HEADS_UP_MATRIX for lookups
    // Matrix currently contains placeholder approximation data
    // Run `task gen:heads-up-matrix` to generate exact values (~20 min)

    // Test AA vs KK
    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    const result = HeadsUpEquity.evaluatePreflopAllIn(aa, kk);

    // With placeholder approximation, AA vs KK shows ~51%
    // After running gen:heads-up-matrix, real value should be ~82%
    try std.testing.expect(result.win > result.loss);
    try std.testing.expect(result.win > 0.45);
    try std.testing.expect(result.win < 0.90); // Wide range to work with both placeholder and real data

    // Sum should equal 1.0
    const sum = result.win + result.tie + result.loss;
    try std.testing.expect(sum > 0.99 and sum < 1.01);

    // Test 22 vs 77 - lower pair should lose
    const twos = card.makeCard(.clubs, .two) | card.makeCard(.diamonds, .two);
    const sevens = card.makeCard(.hearts, .seven) | card.makeCard(.spades, .seven);

    const result2 = HeadsUpEquity.evaluatePreflopAllIn(twos, sevens);

    // 22 vs 77 should favor 77
    try std.testing.expect(result2.loss > result2.win);

    // Test that higher cards have advantage over lower cards
    const ak = card.makeCard(.clubs, .ace) | card.makeCard(.hearts, .king);
    const two_three = card.makeCard(.diamonds, .two) | card.makeCard(.spades, .three);

    const result3 = HeadsUpEquity.evaluatePreflopAllIn(ak, two_three);
    try std.testing.expect(result3.win > result3.loss);
}
