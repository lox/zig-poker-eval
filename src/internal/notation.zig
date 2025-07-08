const std = @import("std");
const card = @import("../card.zig");

// ✅ IMPLEMENTED:
// AsAh       specific cards (ace of spades and ace of hearts)
// AA         pocket pairs (any pair of aces)
// AA, KK     comma-separated ranges (aces or kings)
// JTs        suited combinations (jack-ten suited)
// JTo        offsuit combinations (jack-ten offsuit)
// JT         any combinations (jack-ten suited + offsuit)
// AA, KK, AK mixed ranges (pocket pairs + unpaired hands)
//
// 🚧 TODO - Advanced Notation:
// A*         wildcards (any hand with an ace)
// **         any two cards
// A*s        suited wildcards (any suited ace)
// *h*h       suit patterns (any two hearts)
// X%         percentage ranges (top X% of hands)
// A5-A2      dash ranges (A5,A4,A3,A2)
// AK-JT      connected ranges (AK,KQ,QJ,JT)
// QTs-97s    suited connectors (QTs,J9s,T8s,97s)
// AA-TT      range expressions (pairs tens or higher)

/// Simplified poker notation parsing
pub const NotationError = error{
    InvalidNotation,
    InvalidRank,
    InvalidSuit,
    InvalidLength,
};

/// Parse any poker notation input into hand combinations
/// Handles:
/// - Single notation: "AKo", "AKs", "AK", "88", "AhKs"
/// - Ranges: "AA,KK,QQ", "AK,AQ", "AKs,AKo"
/// - Mixed: "AA,AKo,JJ"
/// Returns all possible hand combinations from the input
pub fn parse(input: []const u8, allocator: std.mem.Allocator) ![]const [2]card.Hand {
    if (input.len < 2) return NotationError.InvalidLength;

    // Handle comma-separated ranges
    if (std.mem.indexOf(u8, input, ",") != null) {
        return parseRange(input, allocator);
    }

    // Handle single notation
    return parseSingle(input, allocator);
}

/// Parse single notation (not comma-separated)
fn parseSingle(notation: []const u8, allocator: std.mem.Allocator) ![]const [2]card.Hand {
    // Handle shorthand notation first (most common case)
    // Handle pocket pairs (e.g., "AA", "KK", "88")
    if (notation.len == 2 and notation[0] == notation[1]) {
        const rank = try parseRankChar(notation[0]);
        return try card.generatePocketPair(rank, allocator);
    }

    // Handle specific cards (e.g., "AhKs", "AdAs")
    if (notation.len == 4) {
        const card1 = parseCard(notation[0..2]) catch return NotationError.InvalidNotation;
        const card2 = parseCard(notation[2..4]) catch return NotationError.InvalidNotation;
        const result = try allocator.alloc([2]card.Hand, 1);
        result[0] = [2]card.Hand{ card1, card2 };
        return result;
    }

    // Handle two different ranks
    if (notation.len >= 2) {
        const rank1 = try parseRankChar(notation[0]);
        const rank2 = try parseRankChar(notation[1]);

        if (rank1 == rank2) {
            // This should be a pocket pair but wasn't caught above
            return try card.generatePocketPair(rank1, allocator);
        }

        // Check for suited/offsuit indicator
        if (notation.len == 2) {
            // No indicator - return all combinations
            return try card.generateAllCombinations(rank1, rank2, allocator);
        } else if (notation.len == 3) {
            const indicator = notation[2];
            switch (indicator) {
                's' => return try card.generateSuitedCombinations(rank1, rank2, allocator),
                'o' => return try card.generateOffsuitCombinations(rank1, rank2, allocator),
                else => return NotationError.InvalidNotation,
            }
        }
    }

    return NotationError.InvalidNotation;
}

/// Parse a rank character to Rank enum
fn parseRankChar(c: u8) !card.Rank {
    return switch (c) {
        '2' => card.Rank.two,
        '3' => card.Rank.three,
        '4' => card.Rank.four,
        '5' => card.Rank.five,
        '6' => card.Rank.six,
        '7' => card.Rank.seven,
        '8' => card.Rank.eight,
        '9' => card.Rank.nine,
        'T', 't' => card.Rank.ten,
        'J', 'j' => card.Rank.jack,
        'Q', 'q' => card.Rank.queen,
        'K', 'k' => card.Rank.king,
        'A', 'a' => card.Rank.ace,
        else => NotationError.InvalidRank,
    };
}

/// Parse a suit character to Suit enum
fn parseSuitChar(c: u8) !card.Suit {
    return switch (c) {
        'h', 'H' => .hearts,
        'd', 'D' => .diamonds,
        'c', 'C' => .clubs,
        's', 'S' => .spades,
        else => NotationError.InvalidSuit,
    };
}

/// Parse a specific card (e.g., "Ah", "Kd")
fn parseCard(input: []const u8) !card.Hand {
    if (input.len != 2) return NotationError.InvalidLength;
    const rank = try parseRankChar(input[0]);
    const suit = try parseSuitChar(input[1]);
    return card.makeCardFromEnums(suit, rank);
}

/// Helper function to extract rank from a single card Hand for testing
/// Returns poker rank (2-14) from a single card's bits
fn getSingleCardRank(single_card: card.Hand) u8 {
    // Check each suit to find which one has the card
    const suits = [4]card.Suit{ .clubs, .diamonds, .hearts, .spades };
    for (suits) |suit| {
        const suit_mask = card.getSuitMask(single_card, suit);
        if (suit_mask != 0) {
            // Find the rank within this suit
            const ranks = [13]card.Rank{ .two, .three, .four, .five, .six, .seven, .eight, .nine, .ten, .jack, .queen, .king, .ace };
            for (ranks, 0..) |rank, rank_idx| {
                if ((suit_mask & (@as(u16, 1) << @intCast(rank_idx))) != 0) {
                    return @intFromEnum(rank) + 2; // Convert to poker rank (2-14)
                }
            }
        }
    }
    return @intFromEnum(card.Rank.two) + 2; // Fallback
}

/// Internal function to parse comma-separated ranges
fn parseRange(range_str: []const u8, allocator: std.mem.Allocator) ![]const [2]card.Hand {
    var hands_list = std.ArrayList([2]card.Hand).init(allocator);
    defer hands_list.deinit();

    var iterator = std.mem.splitSequence(u8, range_str, ",");
    while (iterator.next()) |hand_notation| {
        const trimmed = std.mem.trim(u8, hand_notation, " \t\n\r");
        if (trimmed.len == 0) continue;

        const combinations = try parseSingle(trimmed, allocator);
        defer allocator.free(combinations);

        for (combinations) |combo| {
            try hands_list.append(combo);
        }
    }

    return hands_list.toOwnedSlice();
}

// Tests
test "poker notation parsing" {
    const allocator = std.testing.allocator;

    // Pocket pairs
    const aa = try parse("AA", allocator);
    defer allocator.free(aa);
    try std.testing.expect(aa.len == 6);

    // Suited combinations
    const aks = try parse("AKs", allocator);
    defer allocator.free(aks);
    try std.testing.expect(aks.len == 4);

    // Offsuit combinations
    const ako = try parse("AKo", allocator);
    defer allocator.free(ako);
    try std.testing.expect(ako.len == 12);

    // All combinations
    const ak = try parse("AK", allocator);
    defer allocator.free(ak);
    try std.testing.expect(ak.len == 16);

    // Specific cards
    const specific = try parse("AhKs", allocator);
    defer allocator.free(specific);
    try std.testing.expect(specific.len == 1);
}

test "combination counting (using parse().len)" {
    const allocator = std.testing.allocator;

    const aa = try parse("AA", allocator);
    defer allocator.free(aa);
    try std.testing.expect(aa.len == 6);

    const aks = try parse("AKs", allocator);
    defer allocator.free(aks);
    try std.testing.expect(aks.len == 4);

    const ako = try parse("AKo", allocator);
    defer allocator.free(ako);
    try std.testing.expect(ako.len == 12);

    const ak = try parse("AK", allocator);
    defer allocator.free(ak);
    try std.testing.expect(ak.len == 16);

    const specific = try parse("AhKs", allocator);
    defer allocator.free(specific);
    try std.testing.expect(specific.len == 1);
}

test "range parsing" {
    const allocator = std.testing.allocator;

    // Single hands (no comma)
    const ak = try parse("AK", allocator);
    defer allocator.free(ak);
    try std.testing.expect(ak.len == 16);

    const aa = try parse("AA", allocator);
    defer allocator.free(aa);
    try std.testing.expect(aa.len == 6);

    // Multiple hands
    const multi = try parse("AA,KK,QQ", allocator);
    defer allocator.free(multi);
    try std.testing.expect(multi.len == 18); // 6 + 6 + 6

    const mixed = try parse("AA,AKs,AKo", allocator);
    defer allocator.free(mixed);
    try std.testing.expect(mixed.len == 22); // 6 + 4 + 12

    // With spaces
    const spaced = try parse("AA, KK , QQ", allocator);
    defer allocator.free(spaced);
    try std.testing.expect(spaced.len == 18);

    // With empty entries
    const empty = try parse("AA,,KK", allocator);
    defer allocator.free(empty);
    try std.testing.expect(empty.len == 12); // Should skip empty entry
}

test "range counting (using parse().len)" {
    const allocator = std.testing.allocator;

    // Single hands
    const ak = try parse("AK", allocator);
    defer allocator.free(ak);
    try std.testing.expect(ak.len == 16);

    const aa = try parse("AA", allocator);
    defer allocator.free(aa);
    try std.testing.expect(aa.len == 6);

    // Multiple hands
    const multi1 = try parse("AA,KK,QQ", allocator);
    defer allocator.free(multi1);
    try std.testing.expect(multi1.len == 18);

    const multi2 = try parse("AK,AQ", allocator);
    defer allocator.free(multi2);
    try std.testing.expect(multi2.len == 32); // 16 + 16

    const multi3 = try parse("AA,AKs,AKo", allocator);
    defer allocator.free(multi3);
    try std.testing.expect(multi3.len == 22); // 6 + 4 + 12

    // With spaces
    const spaced = try parse("AA, KK , QQ", allocator);
    defer allocator.free(spaced);
    try std.testing.expect(spaced.len == 18);
}

test "parse and pick random (user pattern)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(12345);
    const rng = prng.random();

    // Users can easily get random hands by parsing then picking
    const aa_combos = try parse("AA", allocator);
    defer allocator.free(aa_combos);
    const random_aa = aa_combos[rng.intRangeLessThan(usize, 0, aa_combos.len)];
    const ace_value = @intFromEnum(card.Rank.ace) + 2; // 14
    try std.testing.expect(getSingleCardRank(random_aa[0]) == ace_value); // Both aces
    try std.testing.expect(getSingleCardRank(random_aa[1]) == ace_value);

    // Same for ranges
    const range_combos = try parse("AA,KK,QQ", allocator);
    defer allocator.free(range_combos);
    const random_pair = range_combos[rng.intRangeLessThan(usize, 0, range_combos.len)];
    const rank = getSingleCardRank(random_pair[0]);
    const king_value = @intFromEnum(card.Rank.king) + 2; // 13
    const queen_value = @intFromEnum(card.Rank.queen) + 2; // 12
    try std.testing.expect(rank == ace_value or rank == king_value or rank == queen_value);
    try std.testing.expect(getSingleCardRank(random_pair[0]) == getSingleCardRank(random_pair[1])); // Same rank
}

test "ultra-simple API works for all examples" {
    const allocator = std.testing.allocator;

    // Test that "AK" parses correctly (not just "AKo")
    const ak = try parse("AK", allocator);
    defer allocator.free(ak);
    try std.testing.expect(ak.len == 16); // Both suited and offsuit

    // Test complex ranges
    const r1 = try parse("AA,KK,QQ,AKs", allocator);
    defer allocator.free(r1);
    try std.testing.expect(r1.len == 22); // 6+6+6+4 = 22

    const r2 = try parse("KK,KQo,AJo", allocator);
    defer allocator.free(r2);
    try std.testing.expect(r2.len == 30); // 6+12+12 = 30

    const r3 = try parse("AA,AKs,AKo", allocator);
    defer allocator.free(r3);
    try std.testing.expect(r3.len == 22); // 6+4+12 = 22

    const r4 = try parse("88,AKo,JJ", allocator);
    defer allocator.free(r4);
    try std.testing.expect(r4.len == 24); // 6+12+6 = 24

}
