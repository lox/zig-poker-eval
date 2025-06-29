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

/// Create a card in the packed format
/// suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
/// rank: 0-12 for ranks 2-A
pub fn makeCard(suit: u8, rank: u8) Hand {
    const offset = suit * 13;
    return @as(Hand, 1) << @intCast(offset + rank);
}

/// Create a card using enums for type safety
pub fn makeCardFromEnums(suit: Suit, rank: Rank) Hand {
    return makeCard(@intFromEnum(suit), @intFromEnum(rank));
}

/// Extract rank mask for a specific suit from a hand
pub fn getSuitMask(hand: Hand, suit: Suit) u16 {
    const suit_num: u8 = @intFromEnum(suit);
    const offset: u6 = @intCast(suit_num * 13);
    return @as(u16, @truncate((hand >> offset) & RANK_MASK));
}

/// Check if a hand contains a specific card
pub fn hasCard(hand: Hand, suit: Suit, rank: Rank) bool {
    const card = makeCardFromEnums(suit, rank);
    return (hand & card) != 0;
}

/// Count total number of cards in hand
pub fn countCards(hand: Hand) u8 {
    return @popCount(hand);
}

// Tests for core functionality
const testing = std.testing;

test "card creation and format" {
    // Test basic card creation
    const ace_spades = makeCard(3, 12); // suit=3 (spades), rank=12 (ace)
    const two_clubs = makeCard(0, 0); // suit=0 (clubs), rank=0 (two)

    // Ace of spades should be bit 51 (39 + 12)
    try testing.expect(ace_spades == (@as(u64, 1) << 51));

    // Two of clubs should be bit 0
    try testing.expect(two_clubs == (@as(u64, 1) << 0));
}

test "card creation with enums" {
    const ace_spades = makeCardFromEnums(.spades, .ace);
    const two_clubs = makeCardFromEnums(.clubs, .two);

    try testing.expect(ace_spades == (@as(u64, 1) << 51));
    try testing.expect(two_clubs == (@as(u64, 1) << 0));
}

test "hand operations" {
    const ace_spades = makeCardFromEnums(.spades, .ace);
    const king_spades = makeCardFromEnums(.spades, .king);
    const hand = ace_spades | king_spades;

    try testing.expect(hasCard(hand, .spades, .ace));
    try testing.expect(hasCard(hand, .spades, .king));
    try testing.expect(!hasCard(hand, .hearts, .ace));
    try testing.expect(countCards(hand) == 2);
}

test "hand from individual cards (CardSet approach)" {
    const ace = makeCardFromEnums(.spades, .ace);
    const king = makeCardFromEnums(.hearts, .king);
    const queen = makeCardFromEnums(.diamonds, .queen);

    const hand = ace | king | queen;

    try testing.expect(hasCard(hand, .spades, .ace));
    try testing.expect(hasCard(hand, .hearts, .king));
    try testing.expect(hasCard(hand, .diamonds, .queen));
    try testing.expect(countCards(hand) == 3);
}

test "hole and board combination (CardSet approach)" {
    const hole_ace = makeCardFromEnums(.spades, .ace);
    const hole_king = makeCardFromEnums(.hearts, .king);
    const hole_hand = hole_ace | hole_king;

    const board_queen = makeCardFromEnums(.diamonds, .queen);
    const board_jack = makeCardFromEnums(.clubs, .jack);
    const board_ten = makeCardFromEnums(.spades, .ten);
    const board_hand = board_queen | board_jack | board_ten;

    const final_hand = hole_hand | board_hand;

    try testing.expect(countCards(final_hand) == 5);
    try testing.expect(hasCard(final_hand, .spades, .ace));
    try testing.expect(hasCard(final_hand, .clubs, .jack));
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
