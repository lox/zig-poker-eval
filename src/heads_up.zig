// Specialized Heads-Up Equity Tables
// Pre-computed win frequencies for heads-up (2 player) poker scenarios

// Based on analysis of holdem-hand-evaluator approach
// https://github.com/b-inary/holdem-hand-evaluator

const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");

// There are 169 unique starting hands in Texas Hold'em:
// - 13 pocket pairs (AA, KK, QQ, ..., 22)
// - 78 suited combinations (AKs, AQs, ..., 32s)
// - 78 offsuit combinations (AKo, AQo, ..., 32o)
//
// We represent these as a 13x13 matrix where:
// - Diagonal entries are pocket pairs
// - Upper triangle are suited hands
// - Lower triangle are offsuit hands

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
};

// Pre-computed equity for all 169 starting hands vs random opponent
// Each entry is (win_rate_x1000, tie_rate_x1000) - stored as u16 to save space
// For example, (850, 23) means 85.0% win, 2.3% tie, 12.7% loss
pub const PREFLOP_VS_RANDOM: [169][2]u16 = computePreflopEquities();

// Compute equities at compile time
//
// NOTE: Real equity calculation would require ~437 billion hand evaluations
// (169 hands × 1225 opponents × 2.1M boards). This would make compilation
// take hours/days. Options for production:
//
// 1. Pre-calculate offline: Generate once, save as data file (recommended)
// 2. Monte Carlo sampling: ~10K simulations per hand (slower compilation)
// 3. Use known values: Published equity data from poker literature
//
// Current implementation uses placeholder values for fast compilation.
// TODO: Replace with actual equity data using one of the above methods.
fn computePreflopEquities() [169][2]u16 {
    @setEvalBranchQuota(10_000_000);
    var result: [169][2]u16 = undefined;

    // Placeholder values that approximate real equities
    for (&result, 0..) |*entry, i| {
        // Pocket pairs get progressively stronger
        const is_pair = (i % 14) == 0;
        if (is_pair) {
            const pair_rank = i / 14;
            // AA = ~85%, KK = ~82%, ..., 22 = ~50%
            const win_rate = 850 - (12 - pair_rank) * 30;
            entry.* = .{ win_rate, 5 };
        } else {
            // Non-pairs: suited slightly better than offsuit
            const row = i / 13;
            const col = i % 13;
            const suited = col < row;
            const base_win = 500 + (row + col) * 15;
            const win_rate = if (suited) base_win + 30 else base_win;
            entry.* = .{ @min(win_rate, 850), 20 };
        }
    }

    return result;
}

// Fast heads-up equity calculation using pre-computed tables
pub const HeadsUpEquity = struct {
    pub const Result = struct {
        win: f32,
        tie: f32,
        loss: f32,
    };

    // Get preflop equity for a specific matchup
    pub fn getPreflopEquity(hero_idx: u8, villain_idx: u8) Result {
        // TODO: This should use a full 169x169 table for exact matchups
        // For now, use vs-random approximation
        const hero_eq = PREFLOP_VS_RANDOM[hero_idx];
        const villain_eq = PREFLOP_VS_RANDOM[villain_idx];

        // Simplified calculation - in reality need full enumeration
        const hero_win = @as(f32, @floatFromInt(hero_eq[0])) / 1000.0;
        const villain_win = @as(f32, @floatFromInt(villain_eq[0])) / 1000.0;
        const tie_rate = @as(f32, @floatFromInt(hero_eq[1] + villain_eq[1])) / 2000.0;

        // Normalize
        const total = hero_win + villain_win;
        if (total > 0) {
            return .{
                .win = hero_win / total * (1.0 - tie_rate),
                .tie = tie_rate,
                .loss = villain_win / total * (1.0 - tie_rate),
            };
        } else {
            return .{ .win = 0.333, .tie = 0.334, .loss = 0.333 };
        }
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
