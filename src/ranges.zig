const std = @import("std");
const poker = @import("poker.zig");

// Generate all pocket pairs (AA, KK, QQ, ..., 22)
pub fn generatePocketPairs(allocator: std.mem.Allocator) ![][2]poker.Card {
    var pairs = std.ArrayList([2]poker.Card).init(allocator);
    defer pairs.deinit();

    // Generate pairs from Aces down to Deuces
    var rank: u8 = 14;
    while (rank >= 2) : (rank -= 1) {
        try pairs.append([_]poker.Card{
            poker.Card.init(rank, 0), // hearts
            poker.Card.init(rank, 1), // spades
        });
    }

    return pairs.toOwnedSlice();
}

// Generate all suited aces (AsKs, AsQs, ..., As2s)
pub fn generateSuitedAces(allocator: std.mem.Allocator) ![][2]poker.Card {
    var suited_aces = std.ArrayList([2]poker.Card).init(allocator);
    defer suited_aces.deinit();

    // Generate suited aces from AK down to A2
    for (2..14) |rank| {
        const r = @as(u8, @intCast(rank));
        try suited_aces.append([_]poker.Card{
            poker.Card.init(14, 0), // Ah
            poker.Card.init(r, 0), // Same suit
        });
    }

    return suited_aces.toOwnedSlice();
}

// Generate all offsuit aces (AhKs, AhQs, ..., AhKd, etc.)
pub fn generateOffsuitAces(allocator: std.mem.Allocator) ![][2]poker.Card {
    var offsuit_aces = std.ArrayList([2]poker.Card).init(allocator);
    defer offsuit_aces.deinit();

    // Generate offsuit aces
    for (2..14) |rank| {
        const r = @as(u8, @intCast(rank));
        // For each rank, generate all suit combinations where suits differ
        inline for (0..4) |ace_suit| {
            inline for (0..4) |kicker_suit| {
                if (ace_suit != kicker_suit) {
                    try offsuit_aces.append([_]poker.Card{
                        poker.Card.init(14, @intCast(ace_suit)),
                        poker.Card.init(r, @intCast(kicker_suit)),
                    });
                }
            }
        }
    }

    return offsuit_aces.toOwnedSlice();
}

// Generate premium starting hands (AA, KK, QQ, JJ, AKs, AQs, AJs, AKo)
pub fn generatePremiumHands(allocator: std.mem.Allocator) ![][2]poker.Card {
    var premium = std.ArrayList([2]poker.Card).init(allocator);
    defer premium.deinit();

    // Premium pocket pairs
    const premium_pairs = [_]u8{ 14, 13, 12, 11 }; // AA, KK, QQ, JJ
    for (premium_pairs) |rank| {
        try premium.append([_]poker.Card{
            poker.Card.init(rank, 0),
            poker.Card.init(rank, 1),
        });
    }

    // Premium suited aces
    const premium_aces_suited = [_]u8{ 13, 12, 11 }; // AKs, AQs, AJs
    for (premium_aces_suited) |rank| {
        try premium.append([_]poker.Card{
            poker.Card.init(14, 0), // As
            poker.Card.init(rank, 0), // Same suit
        });
    }

    // AKo (all offsuit AK combinations)
    inline for (0..4) |ace_suit| {
        inline for (0..4) |king_suit| {
            if (ace_suit != king_suit) {
                try premium.append([_]poker.Card{
                    poker.Card.init(14, @intCast(ace_suit)),
                    poker.Card.init(13, @intCast(king_suit)),
                });
            }
        }
    }

    return premium.toOwnedSlice();
}

// Generate all hands of a specific rank (e.g., all AK combinations)
pub fn generateHandRank(high_rank: u8, low_rank: u8, suited_only: bool, allocator: std.mem.Allocator) ![][2]poker.Card {
    var hands = std.ArrayList([2]poker.Card).init(allocator);
    defer hands.deinit();

    if (high_rank == low_rank) {
        // Pocket pair - only one combination per suit pair
        inline for (0..4) |suit1| {
            inline for (suit1 + 1..4) |suit2| {
                try hands.append([_]poker.Card{
                    poker.Card.init(high_rank, @intCast(suit1)),
                    poker.Card.init(low_rank, @intCast(suit2)),
                });
            }
        }
    } else {
        // Non-pair hand
        if (suited_only) {
            // Only suited combinations
            inline for (0..4) |suit| {
                try hands.append([_]poker.Card{
                    poker.Card.init(high_rank, @intCast(suit)),
                    poker.Card.init(low_rank, @intCast(suit)),
                });
            }
        } else {
            // All combinations (suited and offsuit)
            inline for (0..4) |suit1| {
                inline for (0..4) |suit2| {
                    try hands.append([_]poker.Card{
                        poker.Card.init(high_rank, @intCast(suit1)),
                        poker.Card.init(low_rank, @intCast(suit2)),
                    });
                }
            }
        }
    }

    return hands.toOwnedSlice();
}

// Simple range parser for basic notation (e.g., "AA,KK,QQ,AKs,AKo")
pub fn parseRange(range_str: []const u8, allocator: std.mem.Allocator) ![][2]poker.Card {
    var range = std.ArrayList([2]poker.Card).init(allocator);
    defer range.deinit();

    var it = std.mem.splitSequence(u8, range_str, ",");
    while (it.next()) |hand_str| {
        const trimmed = std.mem.trim(u8, hand_str, " ");
        if (trimmed.len == 0) continue;

        const hands = try parseHandNotation(trimmed, allocator);
        defer allocator.free(hands);

        try range.appendSlice(hands);
    }

    return range.toOwnedSlice();
}

// Parse individual hand notation (AA, AKs, AKo, etc.)
fn parseHandNotation(notation: []const u8, allocator: std.mem.Allocator) ![][2]poker.Card {
    if (notation.len < 2 or notation.len > 3) {
        return error.InvalidNotation;
    }

    // Parse ranks
    const rank1 = parseRankChar(notation[0]);
    const rank2 = parseRankChar(notation[1]);

    if (rank1 == 0 or rank2 == 0) {
        return error.InvalidRank;
    }

    // Determine if suited/offsuit/pair
    if (notation.len == 2) {
        // Pocket pair (e.g., "AA")
        if (rank1 != rank2) {
            return error.InvalidPairNotation;
        }
        return generateHandRank(rank1, rank2, false, allocator);
    } else if (notation.len == 3) {
        const suit_char = notation[2];
        const high_rank = @max(rank1, rank2);
        const low_rank = @min(rank1, rank2);

        switch (suit_char) {
            's' => return generateHandRank(high_rank, low_rank, true, allocator),
            'o' => {
                // Offsuit - generate all combinations then filter out suited
                const all_hands = try generateHandRank(high_rank, low_rank, false, allocator);
                defer allocator.free(all_hands);

                var offsuit = std.ArrayList([2]poker.Card).init(allocator);
                defer offsuit.deinit();

                for (all_hands) |hand| {
                    if (hand[0].getSuit() != hand[1].getSuit()) {
                        try offsuit.append(hand);
                    }
                }

                return offsuit.toOwnedSlice();
            },
            else => return error.InvalidSuitNotation,
        }
    }

    return error.InvalidNotation;
}

// Parse rank character to numeric value
fn parseRankChar(char: u8) u8 {
    return switch (char) {
        '2'...'9' => char - '0',
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        else => 0,
    };
}

// Helper to create a range from specific hands
pub fn createRange(hands: []const [2]poker.Card, allocator: std.mem.Allocator) ![][2]poker.Card {
    const range = try allocator.alloc([2]poker.Card, hands.len);
    @memcpy(range, hands);
    return range;
}

// Tests
const testing = std.testing;

test "generate pocket pairs" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const pairs = try generatePocketPairs(allocator);
    defer allocator.free(pairs);

    try testing.expect(pairs.len == 13); // 13 different pocket pairs

    // Check that AA is first
    try testing.expect(pairs[0][0].getRank() == 14);
    try testing.expect(pairs[0][1].getRank() == 14);

    // Check that 22 is last
    try testing.expect(pairs[12][0].getRank() == 2);
    try testing.expect(pairs[12][1].getRank() == 2);
}

test "generate premium hands" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const premium = try generatePremiumHands(allocator);
    defer allocator.free(premium);

    // Should have 4 pairs + 3 suited aces + 12 offsuit AK = 19 hands
    try testing.expect(premium.len == 19);
}

test "parse range notation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const range = try parseRange("AA,KK,AKs", allocator);
    defer allocator.free(range);

    // Should have 6 AA combinations + 6 KK combinations + 4 AKs combinations = 16
    try testing.expect(range.len == 16);
}

test "generate hand rank" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test AKs (suited only)
    const aks = try generateHandRank(14, 13, true, allocator);
    defer allocator.free(aks);
    try testing.expect(aks.len == 4); // 4 suits

    // Test AKo (all combinations, then filter)
    const ak_all = try generateHandRank(14, 13, false, allocator);
    defer allocator.free(ak_all);
    try testing.expect(ak_all.len == 16); // 4x4 combinations

    // Test AA (pocket pair)
    const aa = try generateHandRank(14, 14, false, allocator);
    defer allocator.free(aa);
    try testing.expect(aa.len == 6); // C(4,2) = 6 combinations
}
