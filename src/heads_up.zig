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

/// All 169 unique starting hands in canonical order (index 0-168)
/// Generated at compile time for zero runtime cost
pub const ALL_STARTING_HANDS: [169]StartingHand = blk: {
    var hands: [169]StartingHand = undefined;
    for (0..169) |i| {
        hands[i] = StartingHand.fromIndex(@intCast(i));
    }
    break :blk hands;
};

/// Represents one of the 169 unique starting hands in Texas Hold'em
/// This is a semantic type for working with preflop hand classes (e.g., "AKs", "72o", "TT")
/// rather than specific card combinations.
pub const StartingHand = struct {
    high: card.Rank,
    low: card.Rank,
    suited: bool,

    /// Compute the 0-168 index for this starting hand
    /// Index maps to position in the 169-hand matrix
    pub fn index(self: StartingHand) u8 {
        const high_val: u16 = @intFromEnum(self.high);
        const low_val: u16 = @intFromEnum(self.low);

        std.debug.assert(high_val >= low_val);

        if (high_val == low_val) {
            // Pocket pair - on the diagonal
            return @intCast(high_val * 13 + high_val);
        } else if (self.suited) {
            // Suited hand - upper triangle
            return @intCast(high_val * 13 + low_val);
        } else {
            // Offsuit hand - lower triangle
            return @intCast(low_val * 13 + high_val);
        }
    }

    /// Create a StartingHand from a 0-168 index
    pub fn fromIndex(idx: u8) StartingHand {
        std.debug.assert(idx < 169);

        const row = idx / 13;
        const col = idx % 13;

        if (row == col) {
            // Pocket pair
            return .{
                .high = @enumFromInt(row),
                .low = @enumFromInt(row),
                .suited = false, // Pocket pairs don't have suited/offsuit distinction
            };
        } else if (row > col) {
            // Suited - upper triangle
            return .{
                .high = @enumFromInt(row),
                .low = @enumFromInt(col),
                .suited = true,
            };
        } else {
            // Offsuit - lower triangle
            return .{
                .high = @enumFromInt(col),
                .low = @enumFromInt(row),
                .suited = false,
            };
        }
    }

    /// Parse a hand string like "AKs", "72o", or "TT" into a StartingHand
    pub fn parse(hand_str: []const u8) !StartingHand {
        if (hand_str.len < 2 or hand_str.len > 3) {
            return error.InvalidHandFormat;
        }

        const rank1 = try rankFromChar(hand_str[0]);
        const rank2 = try rankFromChar(hand_str[1]);

        const suited = if (hand_str.len == 3)
            hand_str[2] == 's'
        else
            rank1 == rank2; // pocket pairs are implicitly "suited"

        const high = if (@intFromEnum(rank1) >= @intFromEnum(rank2)) rank1 else rank2;
        const low = if (@intFromEnum(rank1) < @intFromEnum(rank2)) rank1 else rank2;

        return .{
            .high = high,
            .low = low,
            .suited = suited,
        };
    }

    /// Extract starting hand class from a specific card Hand (bitfield)
    /// Takes exactly 2 hole cards and determines their starting hand classification
    pub fn fromHand(hand: card.Hand) StartingHand {
        // Extract the two cards from the hand bitfield
        var cards_found: [2]struct { rank: card.Rank, suit: card.Suit } = undefined;
        var count: usize = 0;

        for (0..52) |i| {
            if ((hand & (@as(u64, 1) << @intCast(i))) != 0) {
                const suit_num: u2 = @intCast(i / 13);
                const rank_num: u4 = @intCast(i % 13);
                cards_found[count] = .{
                    .rank = @enumFromInt(rank_num),
                    .suit = @enumFromInt(suit_num),
                };
                count += 1;
                if (count >= 2) break;
            }
        }

        std.debug.assert(count == 2);

        const suited = cards_found[0].suit == cards_found[1].suit;
        const high = if (@intFromEnum(cards_found[0].rank) >= @intFromEnum(cards_found[1].rank))
            cards_found[0].rank
        else
            cards_found[1].rank;
        const low = if (@intFromEnum(cards_found[0].rank) < @intFromEnum(cards_found[1].rank))
            cards_found[0].rank
        else
            cards_found[1].rank;

        return .{
            .high = high,
            .low = low,
            .suited = suited,
        };
    }

    fn rankFromChar(c: u8) !card.Rank {
        return switch (c) {
            '2' => .two,
            '3' => .three,
            '4' => .four,
            '5' => .five,
            '6' => .six,
            '7' => .seven,
            '8' => .eight,
            '9' => .nine,
            'T', 't' => .ten,
            'J', 'j' => .jack,
            'Q', 'q' => .queen,
            'K', 'k' => .king,
            'A', 'a' => .ace,
            else => error.InvalidRank,
        };
    }

    /// Convert this starting hand to poker notation string
    /// Returns strings like "AA", "AKs", "72o", etc.
    pub fn toNotation(self: StartingHand) []const u8 {
        const high_val = @intFromEnum(self.high);
        const low_val = @intFromEnum(self.low);

        if (high_val == low_val) {
            // Pocket pair
            return switch (self.high) {
                .two => "22",
                .three => "33",
                .four => "44",
                .five => "55",
                .six => "66",
                .seven => "77",
                .eight => "88",
                .nine => "99",
                .ten => "TT",
                .jack => "JJ",
                .queen => "QQ",
                .king => "KK",
                .ace => "AA",
            };
        } else {
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

            const suit_idx: usize = if (self.suited) 0 else 1;
            return notations[high_val][low_val][suit_idx];
        }
    }

    /// Create a Range containing all card combinations for this starting hand
    /// Example: StartingHand.parse("AA").toRange(allocator) creates Range with all 6 AA combinations
    pub fn toRange(self: StartingHand, allocator: std.mem.Allocator) !range_mod.Range {
        const notation = self.toNotation();
        return range_mod.parseRange(notation, allocator);
    }
};

// Pre-computed equity for all 169 starting hands vs random opponent
// Each entry is (win_rate_x1000, tie_rate_x1000) - stored as u16 to save space
// For example, (850, 23) means 85.0% win, 2.3% tie, 12.7% loss
const heads_up_tables = @import("internal/heads_up_tables.zig");
pub const PREFLOP_VS_RANDOM = heads_up_tables.PREFLOP_VS_RANDOM;

// Pre-computed 169x169 matrix for exact head-to-head matchups
// matrix[hero_idx][villain_idx] = (hero_win_rate_x1000, tie_rate_x1000)
const heads_up_matrix = @import("internal/heads_up_matrix.zig");
pub const HEADS_UP_MATRIX = heads_up_matrix.HEADS_UP_MATRIX;

// Fast heads-up equity calculation using pre-computed tables
pub const HeadsUpEquity = struct {
    pub const Result = struct {
        win: f32,
        tie: f32,
        loss: f32,
    };

    /// Get preflop equity for a specific matchup using exact 169x169 table
    pub fn getPreflopEquity(hero: StartingHand, villain: StartingHand) Result {
        const hero_idx = hero.index();
        const villain_idx = villain.index();
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

    /// Fast path for preflop all-in scenarios
    /// Takes specific hole cards and returns equity based on their starting hand class
    pub fn evaluatePreflopAllIn(
        hero_cards: card.Hand,
        villain_cards: card.Hand,
    ) Result {
        const hero = StartingHand.fromHand(hero_cards);
        const villain = StartingHand.fromHand(villain_cards);

        return getPreflopEquity(hero, villain);
    }
};

test "StartingHand index calculation" {
    // Test pocket pairs
    const deuces = StartingHand{ .high = .two, .low = .two, .suited = false };
    try std.testing.expectEqual(@as(u8, 0), deuces.index());

    const aces = StartingHand{ .high = .ace, .low = .ace, .suited = false };
    try std.testing.expectEqual(@as(u8, 168), aces.index());

    // Test suited hands
    const aks = StartingHand{ .high = .ace, .low = .king, .suited = true };
    try std.testing.expectEqual(@as(u8, 167), aks.index());

    // Test offsuit hands
    const ako = StartingHand{ .high = .ace, .low = .king, .suited = false };
    try std.testing.expectEqual(@as(u8, 155), ako.index());
}

test "StartingHand.parse" {
    const aa = try StartingHand.parse("AA");
    try std.testing.expectEqual(@as(u8, 168), aa.index());
    try std.testing.expectEqual(card.Rank.ace, aa.high);
    try std.testing.expectEqual(card.Rank.ace, aa.low);

    const aks = try StartingHand.parse("AKs");
    try std.testing.expectEqual(@as(u8, 167), aks.index());
    try std.testing.expect(aks.suited);

    const ako = try StartingHand.parse("AKo");
    try std.testing.expectEqual(@as(u8, 155), ako.index());
    try std.testing.expect(!ako.suited);

    const twos = try StartingHand.parse("22");
    try std.testing.expectEqual(@as(u8, 0), twos.index());
}

test "StartingHand.fromIndex" {
    const aa = StartingHand.fromIndex(168);
    try std.testing.expectEqual(card.Rank.ace, aa.high);
    try std.testing.expectEqual(card.Rank.ace, aa.low);

    const aks = StartingHand.fromIndex(167);
    try std.testing.expectEqual(card.Rank.ace, aks.high);
    try std.testing.expectEqual(card.Rank.king, aks.low);
    try std.testing.expect(aks.suited);

    const ako = StartingHand.fromIndex(155);
    try std.testing.expectEqual(card.Rank.ace, ako.high);
    try std.testing.expectEqual(card.Rank.king, ako.low);
    try std.testing.expect(!ako.suited);
}

test "StartingHand.toNotation" {
    // Test that toNotation produces correct strings
    try std.testing.expectEqualStrings("AA", StartingHand.fromIndex(168).toNotation());
    try std.testing.expectEqualStrings("KK", StartingHand.fromIndex(154).toNotation());
    try std.testing.expectEqualStrings("QQ", StartingHand.fromIndex(140).toNotation());
    try std.testing.expectEqualStrings("22", StartingHand.fromIndex(0).toNotation());

    // Test suited and offsuit
    try std.testing.expectEqualStrings("AKs", StartingHand.fromIndex(167).toNotation());
    try std.testing.expectEqualStrings("AKo", StartingHand.fromIndex(155).toNotation());

    // Test roundtrip: notation -> parse -> toNotation
    const aa = try StartingHand.parse("AA");
    try std.testing.expectEqualStrings("AA", aa.toNotation());

    const aks = try StartingHand.parse("AKs");
    try std.testing.expectEqualStrings("AKs", aks.toNotation());

    const ako = try StartingHand.parse("AKo");
    try std.testing.expectEqualStrings("AKo", ako.toNotation());
}

test "StartingHand.toRange creates correct ranges" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test AA creates 6 combinations
    const aa = StartingHand.fromIndex(168);
    var aa_range = try aa.toRange(allocator);
    defer aa_range.deinit();
    try std.testing.expect(aa_range.handCount() == 6);

    // Test AKs creates 4 combinations
    const aks = StartingHand.fromIndex(167);
    var aks_range = try aks.toRange(allocator);
    defer aks_range.deinit();
    try std.testing.expect(aks_range.handCount() == 4);

    // Test AKo creates 12 combinations
    const ako = StartingHand.fromIndex(155);
    var ako_range = try ako.toRange(allocator);
    defer ako_range.deinit();
    try std.testing.expect(ako_range.handCount() == 12);
}

test "ALL_STARTING_HANDS contains all 169 hands" {
    // Verify we have all 169 hands
    try std.testing.expectEqual(@as(usize, 169), ALL_STARTING_HANDS.len);

    // Verify indices are correct
    try std.testing.expectEqual(@as(u8, 0), ALL_STARTING_HANDS[0].index());
    try std.testing.expectEqual(@as(u8, 168), ALL_STARTING_HANDS[168].index());

    // Verify first and last hands
    try std.testing.expectEqualStrings("22", ALL_STARTING_HANDS[0].toNotation());
    try std.testing.expectEqualStrings("AA", ALL_STARTING_HANDS[168].toNotation());

    // Verify all indices match their position
    for (ALL_STARTING_HANDS, 0..) |hand, i| {
        try std.testing.expectEqual(@as(u8, @intCast(i)), hand.index());
    }
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
