const std = @import("std");

// Card and Hand types using the packed u64 format
pub const Hand = u64;

// Card suits
pub const Suit = enum(u2) {
    clubs = 0,
    diamonds = 1,
    hearts = 2,
    spades = 3,
};

// Card ranks (0-12 for 2-A)
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

// Suit offsets in the packed u64 format
pub const CLUBS_OFFSET = 0;
pub const DIAMONDS_OFFSET = 13;
pub const HEARTS_OFFSET = 26;
pub const SPADES_OFFSET = 39;
pub const RANK_MASK = 0x1FFF; // 13 bits

/// Create a card in the packed format
/// suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
/// rank: 0-12 for ranks 2-A
pub fn makeCard(suit: u8, rank: u8) Hand {
    const offset = suit * 13;
    return @as(Hand, 1) << @intCast(offset + rank);
}

/// Create a card using enums
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

/// Create hand from array of individual cards
pub fn makeHandFromCards(cards: []const Hand) Hand {
    var hand: Hand = 0;
    for (cards) |card| {
        hand |= card;
    }
    return hand;
}

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

    return makeCard(suit, rank);
}

/// Parse multiple cards from string like "AsKsQs"
pub fn parseCards(card_string: []const u8, allocator: std.mem.Allocator) ![]Hand {
    if (card_string.len % 2 != 0) {
        return error.InvalidCardString;
    }

    const card_count = card_string.len / 2;
    var cards = try allocator.alloc(Hand, card_count);

    for (0..card_count) |i| {
        const start = i * 2;
        cards[i] = try parseCard(card_string[start..start + 2]);
    }

    return cards;
}

/// Compile-time card parsing - returns fixed array, no allocation needed
pub fn mustParseCards(comptime card_string: []const u8) [card_string.len / 2]Hand {
    if (card_string.len % 2 != 0) {
        @compileError("Invalid card string length: " ++ card_string);
    }

    const card_count = card_string.len / 2;
    var cards: [card_count]Hand = undefined;

    comptime var i: usize = 0;
    inline while (i < card_count) : (i += 1) {
        const start = i * 2;
        const rank_char = card_string[start];
        const suit_char = card_string[start + 1];

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

        cards[i] = makeCard(suit, rank);
    }

    return cards;
}

/// Create a 7-card hand from hole cards and board cards
pub fn makeHandFromHoleAndBoard(hole: [2]Hand, board: []const Hand) Hand {
    var hand = hole[0] | hole[1];
    for (board) |card| {
        hand |= card;
    }
    return hand;
}

/// Create hand from hole cards (as single Hand) and board cards 
pub fn fromHoleAndBoard(hole: [2]Hand, board: []const Hand) Hand {
    return makeHandFromHoleAndBoard(hole, board);
}

/// Create hand from hole bits (as u64) and board bits (as u64)
pub fn fromHoleAndBoardBits(hole_bits: u64, board_bits: u64) Hand {
    return hole_bits | board_bits;
}

/// Generate all suited combinations for a rank pair (4 combinations)
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 4);
    const suits = [_]Suit{ .clubs, .diamonds, .hearts, .spades };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]Hand{
            makeCardFromEnums(suit, rank1),
            makeCardFromEnums(suit, rank2),
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
                    makeCardFromEnums(suit1, rank1),
                    makeCardFromEnums(suit2, rank2),
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
                makeCardFromEnums(suit1, rank1),
                makeCardFromEnums(suit2, rank2),
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
                makeCardFromEnums(suit1, rank),
                makeCardFromEnums(suit2, rank),
            };
            idx += 1;
        }
    }

    return combinations;
}

// Helper types for poker module compatibility
pub const Card = struct {
    bits: u64,

    pub fn init(rank: u8, suit: u2) Card {
        // Convert poker rank (2-14) to card rank (0-12)
        const card_rank = rank - 2;
        return Card{ .bits = makeCard(suit, card_rank) };
    }

    pub fn getRank(self: Card) u8 {
        // Find which bit is set and convert back to poker rank (2-14)
        const card_bits = self.bits;
        
        // Check each suit
        for (0..4) |suit| {
            const suit_mask = getSuitMask(card_bits, @enumFromInt(suit));
            if (suit_mask != 0) {
                // Find the rank within this suit
                for (0..13) |rank| {
                    if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                        return @intCast(rank + 2); // Convert back to poker rank
                    }
                }
            }
        }
        return 2; // Fallback
    }

    pub fn getSuit(self: Card) u2 {
        const card_bits = self.bits;
        
        // Check each suit to find which one has a card
        for (0..4) |suit| {
            const suit_mask = getSuitMask(card_bits, @enumFromInt(suit));
            if (suit_mask != 0) {
                return @intCast(suit);
            }
        }
        return 0; // Fallback
    }
};

// Convenience functions
pub fn createCard(suit: Suit, rank: Rank) Hand {
    return makeCardFromEnums(suit, rank);
}

pub fn createHand(cards: []const struct { Suit, Rank }) Hand {
    var hand: Hand = 0;
    for (cards) |card_info| {
        const card = createCard(card_info[0], card_info[1]);
        hand |= card;
    }
    return hand;
}

pub fn mustParseHoleCards(comptime card_string: []const u8) Hand {
    if (card_string.len != 4) {
        @compileError("Hole cards must be exactly 4 characters (e.g., 'AhAs'): " ++ card_string);
    }
    const cards = mustParseCards(card_string);
    return cards[0] | cards[1];
}

/// Evaluate a hand using the fast evaluator
pub fn evaluate(hand: Hand) u16 {
    const evaluator = @import("../evaluator/mod.zig");
    return evaluator.evaluateHand(hand);
}

/// Convert a single-card Hand to a Card (for backward compatibility)
pub fn handToCard(hand: Hand) Card {
    return Card{ .bits = hand };
}

// Tests
const testing = std.testing;

test "card creation and format" {
    // Test basic card creation
    const ace_spades = makeCard(3, 12); // suit=3 (spades), rank=12 (ace)
    const two_clubs = makeCard(0, 0);   // suit=0 (clubs), rank=0 (two)
    
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

test "poker module compatibility" {
    // Test old Card API for backward compatibility
    const ace_spades = Card.init(14, 1); // rank=14 (ace), suit=1 (spades)
    
    try testing.expect(ace_spades.getRank() == 14);
    try testing.expect(ace_spades.getSuit() == 1);
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

test "card parsing" {
    const ace_spades = try parseCard("As");
    const two_clubs = try parseCard("2c");
    
    try testing.expect(ace_spades == makeCardFromEnums(.spades, .ace));
    try testing.expect(two_clubs == makeCardFromEnums(.clubs, .two));
}