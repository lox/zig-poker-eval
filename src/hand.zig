const std = @import("std");
const card = @import("card");

/// Hand parsing, combination generation, and poker-specific hand utilities
/// This module bridges between raw card bitfields and poker domain concepts

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;

// Note: parseCard functionality moved to card.zig module
// Use card.parseCard for compile-time parsing
// Use card.maybeParseCard for runtime parsing with error handling

/// Compile-time parsing of any card string into a Hand (CardSet)
/// Examples:
///   parseHand("As") → Hand with 1 card
///   parseHand("AsKd") → Hand with 2 cards (hole cards)
///   parseHand("AsKdQh") → Hand with 3 cards (flop)
///   parseHand("AsKdQhJsTs5h2c") → Hand with 7 cards (full hand)
pub fn parseHand(comptime card_string: []const u8) Hand {
    if (card_string.len % 2 != 0) {
        @compileError("Invalid card string length: " ++ card_string);
    }

    var hand: Hand = 0;
    comptime var i: usize = 0;
    inline while (i < card_string.len) : (i += 2) {
        const single_card = card.parseCard(card_string[i .. i + 2]);
        hand |= single_card;
    }
    return hand;
}

/// Runtime parsing of any card string into a Hand (CardSet)
/// Returns an error if the card string is invalid
/// Examples:
///   try maybeParseHand("As") → Hand with 1 card
///   try maybeParseHand("AsKd") → Hand with 2 cards (hole cards)
///   try maybeParseHand("AsKdQhJsTs5h2c") → Hand with 7 cards (full hand)
pub fn maybeParseHand(card_string: []const u8) !Hand {
    if (card_string.len % 2 != 0) {
        return error.InvalidCardStringLength;
    }

    var hand: Hand = 0;
    var i: usize = 0;
    while (i < card_string.len) : (i += 2) {
        const single_card = try card.maybeParseCard(card_string[i .. i + 2]);
        hand |= single_card;
    }
    return hand;
}

/// Generate all suited combinations for a rank pair (4 combinations)
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 4);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]Hand{
            card.makeCard(suit, rank1),
            card.makeCard(suit, rank2),
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
                    card.makeCard(suit1, rank1),
                    card.makeCard(suit2, rank2),
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
                card.makeCard(suit1, rank1),
                card.makeCard(suit2, rank2),
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
                card.makeCard(suit1, rank),
                card.makeCard(suit2, rank),
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
        const card_bits = card.makeCard(card_info[0], card_info[1]);
        hand |= card_bits;
    }
    return hand;
}

/// Check if two hole hands and/or board have any conflicting cards
pub fn hasCardConflict(hero_hole: [2]Hand, villain_hole: [2]Hand, board: []const Hand) bool {
    // Combine cards into hands using bitwise OR
    const hero = hero_hole[0] | hero_hole[1];
    const villain = villain_hole[0] | villain_hole[1];
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    // Check for any overlapping bits
    return (hero & villain) != 0 or
        (hero & board_hand) != 0 or
        (villain & board_hand) != 0;
}

// Tests for hand parsing and generation
const testing = std.testing;

test "card parsing" {
    const ace_spades = try card.maybeParseCard("As");
    const two_clubs = try card.maybeParseCard("2c");

    try testing.expect(ace_spades == card.makeCard(.spades, .ace));
    try testing.expect(two_clubs == card.makeCard(.clubs, .two));
}

test "runtime single card parsing" {
    // Test individual card parsing for user input
    const ace = try card.maybeParseCard("As");
    const king = try card.maybeParseCard("Kh");
    const queen = try card.maybeParseCard("Qd");

    try testing.expect(ace == card.makeCard(.spades, .ace));
    try testing.expect(king == card.makeCard(.hearts, .king));
    try testing.expect(queen == card.makeCard(.diamonds, .queen));

    // Test that individual cards can be combined into hands
    const combined = ace | king | queen;
    try testing.expect(card.countCards(combined) == 3);
    try testing.expect(card.hasCard(combined, .spades, .ace));
    try testing.expect(card.hasCard(combined, .hearts, .king));
    try testing.expect(card.hasCard(combined, .diamonds, .queen));
}

test "compile-time hand parsing" {
    // Test single card
    const single_card = parseHand("As");
    try testing.expect(card.hasCard(single_card, .spades, .ace));
    try testing.expect(card.countCards(single_card) == 1);

    // Test multiple cards
    const three_cards = parseHand("AsKhQd");
    try testing.expect(card.hasCard(three_cards, .spades, .ace));
    try testing.expect(card.hasCard(three_cards, .hearts, .king));
    try testing.expect(card.hasCard(three_cards, .diamonds, .queen));
    try testing.expect(card.countCards(three_cards) == 3);

    // Test hole cards
    const hole = parseHand("AsKh");
    try testing.expect(card.hasCard(hole, .spades, .ace));
    try testing.expect(card.hasCard(hole, .hearts, .king));
    try testing.expect(card.countCards(hole) == 2);

    // Test full hand
    const full_hand = parseHand("AsKhQdJcTs5h2d");
    try testing.expect(card.countCards(full_hand) == 7);
}

test "runtime hand parsing" {
    // Test single card
    const single_card = try maybeParseHand("As");
    try testing.expect(card.hasCard(single_card, .spades, .ace));
    try testing.expect(card.countCards(single_card) == 1);

    // Test multiple cards
    const three_cards = try maybeParseHand("AsKhQd");
    try testing.expect(card.hasCard(three_cards, .spades, .ace));
    try testing.expect(card.hasCard(three_cards, .hearts, .king));
    try testing.expect(card.hasCard(three_cards, .diamonds, .queen));
    try testing.expect(card.countCards(three_cards) == 3);

    // Test hole cards
    const hole = try maybeParseHand("AsKh");
    try testing.expect(card.hasCard(hole, .spades, .ace));
    try testing.expect(card.hasCard(hole, .hearts, .king));
    try testing.expect(card.countCards(hole) == 2);

    // Test full hand
    const full_hand = try maybeParseHand("AsKhQdJcTs5h2d");
    try testing.expect(card.countCards(full_hand) == 7);

    // Test error cases
    const invalid_length = maybeParseHand("AsK");
    try testing.expectError(error.InvalidCardStringLength, invalid_length);

    const invalid_card = maybeParseHand("AsXx");
    try testing.expectError(error.InvalidRank, invalid_card);
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
    const ace_spades = card.makeCard(.spades, .ace);
    try testing.expect(ace_spades == card.makeCard(.spades, .ace));

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

test "hasCardConflict detects overlapping cards" {
    // Conflict in hole cards
    const hero1 = [2]Hand{
        card.makeCard(.clubs, .ace),
        card.makeCard(.diamonds, .ace),
    };
    const villain1 = [2]Hand{
        card.makeCard(.clubs, .ace), // Same as hero
        card.makeCard(.hearts, .king),
    };
    try testing.expect(hasCardConflict(hero1, villain1, &.{}));

    // Conflict with board
    const hero2 = [2]Hand{
        card.makeCard(.spades, .ace),
        card.makeCard(.spades, .king),
    };
    const villain2 = [2]Hand{
        card.makeCard(.hearts, .ace),
        card.makeCard(.hearts, .king),
    };
    const board = [_]Hand{
        card.makeCard(.spades, .ace), // Same as hero's first card
        card.makeCard(.diamonds, .seven),
        card.makeCard(.clubs, .two),
    };
    try testing.expect(hasCardConflict(hero2, villain2, &board));

    // No conflicts
    const hero3 = [2]Hand{
        card.makeCard(.clubs, .ace),
        card.makeCard(.clubs, .king),
    };
    const villain3 = [2]Hand{
        card.makeCard(.hearts, .queen),
        card.makeCard(.hearts, .jack),
    };
    const clean_board = [_]Hand{
        card.makeCard(.diamonds, .ten),
        card.makeCard(.spades, .nine),
        card.makeCard(.clubs, .eight),
    };
    try testing.expect(!hasCardConflict(hero3, villain3, &clean_board));
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
