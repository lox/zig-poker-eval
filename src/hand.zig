const std = @import("std");
const card = @import("card");

/// Hand parsing, combination generation, and poker-specific hand utilities
/// This module bridges between raw card bitfields and poker domain concepts

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;

/// Parse card string like "As" into a card
/// rank: '2'-'9', 'T', 'J', 'Q', 'K', 'A'
/// suit: 'c'=clubs, 'd'=diamonds, 'h'=hearts, 's'=spades
pub fn parseCard(card_str: []const u8) !Hand {
    if (card_str.len != 2) return error.InvalidCardString;

    // Parse rank
    const rank: u8 = switch (card_str[0]) {
        '2'...'9' => card_str[0] - '2',
        'T' => 8,
        'J' => 9,
        'Q' => 10,
        'K' => 11,
        'A' => 12,
        else => return error.InvalidRank,
    };

    // Parse suit
    const suit: u8 = switch (card_str[1]) {
        'c' => 0, // clubs
        'd' => 1, // diamonds
        'h' => 2, // hearts
        's' => 3, // spades
        else => return error.InvalidSuit,
    };

    return card.makeCard(suit, rank);
}

/// Compile-time parsing of any card string into a Hand (CardSet)
/// Examples:
///   mustParseHand("As") → Hand with 1 card
///   mustParseHand("AsKd") → Hand with 2 cards (hole cards)
///   mustParseHand("AsKdQh") → Hand with 3 cards (flop)
///   mustParseHand("AsKdQhJsTs5h2c") → Hand with 7 cards (full hand)
pub fn mustParseHand(comptime card_string: []const u8) Hand {
    if (card_string.len % 2 != 0) {
        @compileError("Invalid card string length: " ++ card_string);
    }

    var hand: Hand = 0;
    comptime var i: usize = 0;
    inline while (i < card_string.len) : (i += 2) {
        const single_card = mustParseCard(card_string[i .. i + 2]);
        hand |= single_card;
    }
    return hand;
}

/// Compile-time single card parsing helper
fn mustParseCard(comptime card_str: []const u8) Hand {
    if (card_str.len != 2) {
        @compileError("Card must be exactly 2 characters: " ++ card_str);
    }

    const rank_char = card_str[0];
    const suit_char = card_str[1];

    // Parse rank
    const rank: u8 = switch (rank_char) {
        '2'...'9' => rank_char - '2',
        'T' => 8,
        'J' => 9,
        'Q' => 10,
        'K' => 11,
        'A' => 12,
        else => @compileError("Invalid rank: " ++ [_]u8{rank_char}),
    };

    // Parse suit
    const suit: u8 = switch (suit_char) {
        'c' => 0, // clubs
        'd' => 1, // diamonds
        'h' => 2, // hearts
        's' => 3, // spades
        else => @compileError("Invalid suit: " ++ [_]u8{suit_char}),
    };

    return card.makeCard(suit, rank);
}

/// Generate all suited combinations for a rank pair (4 combinations)
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 4);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]Hand{
            card.makeCardFromEnums(suit, rank1),
            card.makeCardFromEnums(suit, rank2),
        };
    }

    return combinations;
}

/// Generate all offsuit combinations for a rank pair (12 combinations)
pub fn generateOffsuitCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 12);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };
    var idx: usize = 0;

    for (suits) |suit1| {
        for (suits) |suit2| {
            if (suit1 != suit2) {
                combinations[idx] = [2]Hand{
                    card.makeCardFromEnums(suit1, rank1),
                    card.makeCardFromEnums(suit2, rank2),
                };
                idx += 1;
            }
        }
    }

    return combinations;
}

/// Generate all combinations (suited + offsuit = 16 combinations)
pub fn generateAllCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 16);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };
    var idx: usize = 0;

    for (suits) |suit1| {
        for (suits) |suit2| {
            combinations[idx] = [2]Hand{
                card.makeCardFromEnums(suit1, rank1),
                card.makeCardFromEnums(suit2, rank2),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Generate all combinations for a pocket pair (6 combinations)
pub fn generatePocketPair(rank: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 6);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };
    var idx: usize = 0;

    for (suits, 0..) |suit1, i| {
        for (suits[i + 1 ..]) |suit2| {
            combinations[idx] = [2]Hand{
                card.makeCardFromEnums(suit1, rank),
                card.makeCardFromEnums(suit2, rank),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Convenience function for creating hands from suit/rank pairs
pub fn createHand(cards: []const struct { Suit, Rank }) Hand {
    var hand: Hand = 0;
    for (cards) |card_info| {
        const card_bits = card.makeCardFromEnums(card_info[0], card_info[1]);
        hand |= card_bits;
    }
    return hand;
}

// Tests for hand parsing and generation
const testing = std.testing;

test "card parsing" {
    const ace_spades = try parseCard("As");
    const two_clubs = try parseCard("2c");

    try testing.expect(ace_spades == card.makeCardFromEnums(.spades, .ace));
    try testing.expect(two_clubs == card.makeCardFromEnums(.clubs, .two));
}

test "runtime single card parsing" {
    // Test individual card parsing for user input
    const ace = try parseCard("As");
    const king = try parseCard("Kh");
    const queen = try parseCard("Qd");

    try testing.expect(ace == card.makeCardFromEnums(.spades, .ace));
    try testing.expect(king == card.makeCardFromEnums(.hearts, .king));
    try testing.expect(queen == card.makeCardFromEnums(.diamonds, .queen));

    // Test that individual cards can be combined into hands
    const combined = ace | king | queen;
    try testing.expect(card.countCards(combined) == 3);
    try testing.expect(card.hasCard(combined, .spades, .ace));
    try testing.expect(card.hasCard(combined, .hearts, .king));
    try testing.expect(card.hasCard(combined, .diamonds, .queen));
}

test "compile-time hand parsing" {
    // Test single card
    const single_card = mustParseHand("As");
    try testing.expect(card.hasCard(single_card, .spades, .ace));
    try testing.expect(card.countCards(single_card) == 1);

    // Test multiple cards
    const three_cards = mustParseHand("AsKhQd");
    try testing.expect(card.hasCard(three_cards, .spades, .ace));
    try testing.expect(card.hasCard(three_cards, .hearts, .king));
    try testing.expect(card.hasCard(three_cards, .diamonds, .queen));
    try testing.expect(card.countCards(three_cards) == 3);

    // Test hole cards
    const hole = mustParseHand("AsKh");
    try testing.expect(card.hasCard(hole, .spades, .ace));
    try testing.expect(card.hasCard(hole, .hearts, .king));
    try testing.expect(card.countCards(hole) == 2);

    // Test full hand
    const full_hand = mustParseHand("AsKhQdJcTs5h2d");
    try testing.expect(card.countCards(full_hand) == 7);
}

test "suited combinations generation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const combinations = try generateSuitedCombinations(.ace, .king, allocator);
    defer allocator.free(combinations);

    try testing.expect(combinations.len == 4);

    // Each combination should be suited
    for (combinations) |combo| {
        const hand = combo[0] | combo[1];
        // Both cards should be aces and kings
        try testing.expect(card.hasCard(hand, .clubs, .ace) or card.hasCard(hand, .diamonds, .ace) or card.hasCard(hand, .hearts, .ace) or card.hasCard(hand, .spades, .ace));
        try testing.expect(card.hasCard(hand, .clubs, .king) or card.hasCard(hand, .diamonds, .king) or card.hasCard(hand, .hearts, .king) or card.hasCard(hand, .spades, .king));
    }
}

test "pocket pair generation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const combinations = try generatePocketPair(.ace, allocator);
    defer allocator.free(combinations);

    try testing.expect(combinations.len == 6); // C(4,2) = 6 combinations

    // Each combination should be a pair of aces
    for (combinations) |combo| {
        const hand = combo[0] | combo[1];
        try testing.expect(card.countCards(hand) == 2);

        // Count aces in each suit
        var ace_count: u8 = 0;
        if (card.hasCard(hand, .clubs, .ace)) ace_count += 1;
        if (card.hasCard(hand, .diamonds, .ace)) ace_count += 1;
        if (card.hasCard(hand, .hearts, .ace)) ace_count += 1;
        if (card.hasCard(hand, .spades, .ace)) ace_count += 1;

        try testing.expect(ace_count == 2);
    }
}

test "convenience functions" {
    const ace_spades = card.makeCardFromEnums(.spades, .ace);
    try testing.expect(ace_spades == card.makeCardFromEnums(.spades, .ace));

    const hand = createHand(&.{
        .{ .spades, .ace },
        .{ .hearts, .king },
        .{ .diamonds, .queen },
    });
    try testing.expect(card.hasCard(hand, .spades, .ace));
    try testing.expect(card.hasCard(hand, .hearts, .king));
    try testing.expect(card.hasCard(hand, .diamonds, .queen));
    try testing.expect(card.countCards(hand) == 3);
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
