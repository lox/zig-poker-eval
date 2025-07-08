const std = @import("std");

/// Core card representation using packed u64 format
/// Each card is represented as a single bit in a 64-bit integer
/// Layout: [13 spades][13 hearts][13 diamonds][13 clubs]
/// This allows for extremely fast bitwise operations
pub const Hand = u64;

/// Card suits (2-bit enum)
pub const Suit = enum(u2) {
    clubs = 0,
    diamonds = 1,
    hearts = 2,
    spades = 3,
};

/// Card ranks (4-bit enum, 0-12 for 2-A)
pub const Rank = enum(u4) {
    two = 0,
    three = 1,
    four = 2,
    five = 3,
    six = 4,
    seven = 5,
    eight = 6,
    nine = 7,
    ten = 8,
    jack = 9,
    queen = 10,
    king = 11,
    ace = 12,
};

/// Bit layout constants for the packed u64 format
pub const CLUBS_OFFSET = 0;
pub const DIAMONDS_OFFSET = 13;
pub const HEARTS_OFFSET = 26;
pub const SPADES_OFFSET = 39;
pub const RANK_MASK = 0x1FFF; // 13 bits for ranks

/// Create a card using enums for type safety
pub fn makeCard(suit: Suit, rank: Rank) Hand {
    const suit_num: u8 = @intFromEnum(suit);
    const rank_num: u8 = @intFromEnum(rank);
    const offset = suit_num * 13;
    return @as(Hand, 1) << @intCast(offset + rank_num);
}

/// Extract rank mask for a specific suit from a hand
pub fn getSuitMask(hand: Hand, suit: Suit) u16 {
    const suit_num: u8 = @intFromEnum(suit);
    const offset: u6 = @intCast(suit_num * 13);
    return @as(u16, @truncate((hand >> offset) & RANK_MASK));
}

/// Check if a hand contains a specific card
pub fn hasCard(hand: Hand, suit: Suit, rank: Rank) bool {
    const card = makeCard(suit, rank);
    return (hand & card) != 0;
}

/// Count total number of cards in hand
pub fn countCards(hand: Hand) u8 {
    return @popCount(hand);
}

/// Parse a single card from string notation at compile time
/// Example: parseCard("As") -> Ace of Spades (at compile time)
pub fn parseCard(comptime card_str: []const u8) Hand {
    if (card_str.len != 2) {
        @compileError("Card must be exactly 2 characters: " ++ card_str);
    }

    const rank_char = card_str[0];
    const suit_char = card_str[1];

    // Parse rank
    const rank: Rank = switch (rank_char) {
        '2' => .two,
        '3' => .three,
        '4' => .four,
        '5' => .five,
        '6' => .six,
        '7' => .seven,
        '8' => .eight,
        '9' => .nine,
        'T' => .ten,
        'J' => .jack,
        'Q' => .queen,
        'K' => .king,
        'A' => .ace,
        else => @compileError("Invalid rank character: " ++ [1]u8{rank_char}),
    };

    // Parse suit
    const suit: Suit = switch (suit_char) {
        'c' => .clubs,
        'd' => .diamonds,
        'h' => .hearts,
        's' => .spades,
        else => @compileError("Invalid suit character: " ++ [1]u8{suit_char}),
    };

    return makeCard(suit, rank);
}

/// Parse a single card from string notation at runtime, returning error on invalid input
/// Example: maybeParseCard("As") catch error.InvalidCardString
pub fn maybeParseCard(card_str: []const u8) !Hand {
    if (card_str.len != 2) return error.InvalidCardString;

    // Parse rank
    const rank: Rank = switch (card_str[0]) {
        '2' => .two,
        '3' => .three,
        '4' => .four,
        '5' => .five,
        '6' => .six,
        '7' => .seven,
        '8' => .eight,
        '9' => .nine,
        'T' => .ten,
        'J' => .jack,
        'Q' => .queen,
        'K' => .king,
        'A' => .ace,
        else => return error.InvalidRank,
    };

    // Parse suit
    const suit: Suit = switch (card_str[1]) {
        'c' => .clubs,
        'd' => .diamonds,
        'h' => .hearts,
        's' => .spades,
        else => return error.InvalidSuit,
    };

    return makeCard(suit, rank);
}

/// Format a single card (represented as a bit position) to poker notation
/// Takes a card as a u64 with exactly one bit set and returns 2-character string
/// Example: formatCard(makeCard(.spades, .ace)) -> "As" (Ace of Spades)
pub fn formatCard(card: Hand) [2]u8 {
    // Find which bit is set (there should be exactly one)
    const bit_pos = @ctz(card);

    // Extract suit and rank from bit position
    const suit_num = bit_pos / 13;
    const rank_num = bit_pos % 13;

    // Convert rank to character
    const rank_char: u8 = switch (rank_num) {
        0...8 => '2' + @as(u8, @intCast(rank_num)), // 2-9, T
        9 => 'J',
        10 => 'Q',
        11 => 'K',
        12 => 'A',
        else => '?', // Should never happen
    };

    // Handle '10' -> 'T' case
    const final_rank_char = if (rank_num == 8) 'T' else rank_char;

    // Convert suit to character
    const suit_char: u8 = switch (suit_num) {
        0 => 'c', // clubs
        1 => 'd', // diamonds
        2 => 'h', // hearts
        3 => 's', // spades
        else => '?', // Should never happen
    };

    return [2]u8{ final_rank_char, suit_char };
}

// Tests for core functionality
const testing = std.testing;

test "card creation and format" {
    // Test basic card creation
    const ace_spades = makeCard(.spades, .ace);
    const two_clubs = makeCard(.clubs, .two);

    // Ace of spades should be bit 51 (39 + 12)
    try testing.expect(ace_spades == (@as(u64, 1) << 51));

    // Two of clubs should be bit 0
    try testing.expect(two_clubs == (@as(u64, 1) << 0));
}

test "enum consistency" {
    // Test that enum values match expected bit positions
    const ace_spades = makeCard(.spades, .ace);
    const two_clubs = makeCard(.clubs, .two);

    try testing.expect(ace_spades == (@as(u64, 1) << 51));
    try testing.expect(two_clubs == (@as(u64, 1) << 0));
}

test "hand operations" {
    const ace_spades = makeCard(.spades, .ace);
    const king_spades = makeCard(.spades, .king);
    const hand = ace_spades | king_spades;

    try testing.expect(hasCard(hand, .spades, .ace));
    try testing.expect(hasCard(hand, .spades, .king));
    try testing.expect(!hasCard(hand, .hearts, .ace));
    try testing.expect(countCards(hand) == 2);
}

test "hand from individual cards (CardSet approach)" {
    const ace = makeCard(.spades, .ace);
    const king = makeCard(.hearts, .king);
    const queen = makeCard(.diamonds, .queen);

    const hand = ace | king | queen;

    try testing.expect(hasCard(hand, .spades, .ace));
    try testing.expect(hasCard(hand, .hearts, .king));
    try testing.expect(hasCard(hand, .diamonds, .queen));
    try testing.expect(countCards(hand) == 3);
}

test "hole and board combination (CardSet approach)" {
    const hole_ace = makeCard(.spades, .ace);
    const hole_king = makeCard(.hearts, .king);
    const hole_hand = hole_ace | hole_king;

    const board_queen = makeCard(.diamonds, .queen);
    const board_jack = makeCard(.clubs, .jack);
    const board_ten = makeCard(.spades, .ten);
    const board_hand = board_queen | board_jack | board_ten;

    const final_hand = hole_hand | board_hand;

    try testing.expect(countCards(final_hand) == 5);
    try testing.expect(hasCard(final_hand, .spades, .ace));
    try testing.expect(hasCard(final_hand, .clubs, .jack));
}

test "card formatting" {
    // Test all ranks
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.spades, .ace)), "As");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.hearts, .king)), "Kh");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.diamonds, .queen)), "Qd");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.clubs, .jack)), "Jc");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.spades, .ten)), "Ts");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.hearts, .nine)), "9h");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.diamonds, .two)), "2d");

    // Test all suits with same rank
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.clubs, .ace)), "Ac");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.diamonds, .ace)), "Ad");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.hearts, .ace)), "Ah");
    try testing.expectEqualSlices(u8, &formatCard(makeCard(.spades, .ace)), "As");
}

test "parseCard compile time" {
    // Test parsing matches creating cards
    try testing.expectEqual(parseCard("As"), makeCard(.spades, .ace));
    try testing.expectEqual(parseCard("Kh"), makeCard(.hearts, .king));
    try testing.expectEqual(parseCard("Qd"), makeCard(.diamonds, .queen));
    try testing.expectEqual(parseCard("Jc"), makeCard(.clubs, .jack));
    try testing.expectEqual(parseCard("Ts"), makeCard(.spades, .ten));
    try testing.expectEqual(parseCard("2c"), makeCard(.clubs, .two));

    // Test round-trip: parse and format should match
    try testing.expectEqualSlices(u8, &formatCard(parseCard("As")), "As");
    try testing.expectEqualSlices(u8, &formatCard(parseCard("2c")), "2c");
}

test "maybeParseCard runtime" {
    // Valid cards
    try testing.expectEqual(try maybeParseCard("As"), makeCard(.spades, .ace));
    try testing.expectEqual(try maybeParseCard("2c"), makeCard(.clubs, .two));

    // Invalid cards
    try testing.expectError(error.InvalidCardString, maybeParseCard("A"));
    try testing.expectError(error.InvalidCardString, maybeParseCard("Asd"));
    try testing.expectError(error.InvalidRank, maybeParseCard("Xs"));
    try testing.expectError(error.InvalidSuit, maybeParseCard("Ax"));
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
