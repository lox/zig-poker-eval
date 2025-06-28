const std = @import("std");

// Card suits
pub const Suit = enum(u2) {
    hearts = 0,
    spades = 1,
    diamonds = 2,
    clubs = 3,
};

// Card ranks
pub const Rank = enum(u4) {
    two = 2,
    three = 3,
    four = 4,
    five = 5,
    six = 6,
    seven = 7,
    eight = 8,
    nine = 9,
    ten = 10,
    jack = 11,
    queen = 12,
    king = 13,
    ace = 14,
};

// Hand rankings (from weakest to strongest)
pub const HandRank = enum(u4) {
    high_card = 1,
    pair = 2,
    two_pair = 3,
    three_of_a_kind = 4,
    straight = 5,
    flush = 6,
    full_house = 7,
    four_of_a_kind = 8,
    straight_flush = 9,
};

// Efficient card representation using bit manipulation
pub const Card = struct {
    bits: u64,

    pub fn init(rank: u8, suit: u2) Card {
        // Each card is represented as a single bit
        // Cards 0-51: rank 2-14, suits 0-3
        const card_index = (rank - 2) * 4 + suit;
        return Card{ .bits = @as(u64, 1) << @intCast(card_index) };
    }

    pub fn getRank(self: Card) u8 {
        const card_index = @ctz(self.bits);
        return @intCast((card_index / 4) + 2);
    }

    pub fn getSuit(self: Card) u2 {
        const card_index = @ctz(self.bits);
        return @intCast(card_index % 4);
    }
};

// Hand representation that bridges to our high-performance evaluator
pub const Hand = struct {
    bits: u64,

    pub fn init() Hand {
        return Hand{ .bits = 0 };
    }

    pub fn addCard(self: *Hand, card: Card) void {
        self.bits |= card.bits;
    }

    pub fn fromCards(cards: []const Card) Hand {
        var hand_bits: u64 = 0;
        for (cards) |card| {
            hand_bits |= card.bits;
        }
        return Hand{ .bits = hand_bits };
    }

    pub fn fromHoleAndBoard(hole: [2]Card, board: []const Card) Hand {
        const hole_bits = hole[0].bits | hole[1].bits;
        const board_bits = cardsToBits(board);
        return Hand{ .bits = hole_bits | board_bits };
    }

    pub fn fromHoleAndBoardBits(hole: [2]Card, board_bits: u64) Hand {
        const hole_bits = hole[0].bits | hole[1].bits;
        return Hand{ .bits = hole_bits | board_bits };
    }

    pub fn fromBoard(board: []const Card) Hand {
        return Hand{ .bits = cardsToBits(board) };
    }

    // Bridge to high-performance evaluator - function must be provided
    pub inline fn evaluate(self: Hand, evaluateHandFn: fn(u64) u16) HandRank {
        const rank = evaluateHandFn(self.bits);
        // Convert evaluator rank (lower is better) to our HandRank enum
        return convertEvaluatorRank(rank);
    }

    // Hand composition methods
    pub fn combineWith(self: Hand, other: Hand) Hand {
        return Hand{ .bits = self.bits | other.bits };
    }

    // Check for card conflicts between hands
    pub fn hasConflictWith(self: Hand, other: Hand) bool {
        return (self.bits & other.bits) != 0;
    }

    // Compare two hands for showdown
    pub fn compareWith(self: Hand, other: Hand, evaluateHandFn: fn(u64) u16) ShowdownResult {
        const self_rank = self.evaluate(evaluateHandFn);
        const other_rank = other.evaluate(evaluateHandFn);
        
        if (@intFromEnum(self_rank) > @intFromEnum(other_rank)) {
            return .{ .winner = 0, .tie = false, .winning_rank = self_rank };
        } else if (@intFromEnum(other_rank) > @intFromEnum(self_rank)) {
            return .{ .winner = 1, .tie = false, .winning_rank = other_rank };
        } else {
            return .{ .winner = 0, .tie = true, .winning_rank = self_rank };
        }
    }
};

// Common showdown result type
pub const ShowdownResult = struct { 
    winner: u8, 
    tie: bool, 
    winning_rank: HandRank 
};

// Convert evaluator rank (lower=better) to poker HandRank enum
inline fn convertEvaluatorRank(rank: u16) HandRank {
    // Evaluator uses lower numbers for better hands
    // Our HandRank enum uses higher numbers for better hands
    // This mapping is based on the actual evaluator rank ranges
    if (rank <= 10) return .straight_flush;  // Royal flush + straight flushes
    if (rank <= 166) return .four_of_a_kind;
    if (rank <= 322) return .full_house;
    if (rank <= 1599) return .flush;
    if (rank <= 1609) return .straight;
    if (rank <= 2467) return .three_of_a_kind;
    if (rank <= 3325) return .two_pair;
    if (rank <= 6185) return .pair;
    return .high_card;  // Weakest hands
}

// Convenience functions for creating cards with enums
pub fn createCard(suit: Suit, rank: Rank) Card {
    return Card.init(@intFromEnum(rank), @intFromEnum(suit));
}

pub fn createHand(cards: []const struct { Suit, Rank }) Hand {
    var hand = Hand.init();
    for (cards) |card_info| {
        const card = createCard(card_info[0], card_info[1]);
        hand.addCard(card);
    }
    return hand;
}

// Parse card string like "AsKsQsJsTs2h3h" into slice
pub fn parseCards(card_string: []const u8, allocator: std.mem.Allocator) ![]Card {
    if (card_string.len % 2 != 0) {
        return error.InvalidCardString;
    }

    const card_count = card_string.len / 2;
    var cards = try allocator.alloc(Card, card_count);
    var i: usize = 0;

    while (i < card_string.len) : (i += 2) {
        const rank_char = card_string[i];
        const suit_char = card_string[i + 1];

        // Parse rank
        const rank: u8 = switch (rank_char) {
            '2'...'9' => rank_char - '0',
            'T' => 10,
            'J' => 11,
            'Q' => 12,
            'K' => 13,
            'A' => 14,
            else => return error.InvalidRank,
        };

        // Parse suit
        const suit: u2 = switch (suit_char) {
            'h' => 0, // hearts
            's' => 1, // spades
            'd' => 2, // diamonds
            'c' => 3, // clubs
            else => return error.InvalidSuit,
        };

        cards[i / 2] = Card.init(rank, suit);
    }

    return cards;
}

/// Compile-time card parsing - returns fixed array, no allocation needed
pub fn mustParseCards(comptime card_string: []const u8) [card_string.len / 2]Card {
    if (card_string.len % 2 != 0) {
        @compileError("Invalid card string length: " ++ card_string);
    }

    const card_count = card_string.len / 2;
    var cards: [card_count]Card = undefined;

    comptime var i: usize = 0;
    inline while (i < card_string.len) : (i += 2) {
        const rank_char = card_string[i];
        const suit_char = card_string[i + 1];

        // Parse rank
        const rank: u8 = switch (rank_char) {
            '2'...'9' => rank_char - '0',
            'T' => 10,
            'J' => 11,
            'Q' => 12,
            'K' => 13,
            'A' => 14,
            else => @compileError("Invalid rank: " ++ [_]u8{rank_char}),
        };

        // Parse suit
        const suit: u2 = switch (suit_char) {
            'h' => 0, // hearts
            's' => 1, // spades
            'd' => 2, // diamonds
            'c' => 3, // clubs
            else => @compileError("Invalid suit: " ++ [_]u8{suit_char}),
        };

        cards[i / 2] = Card.init(rank, suit);
    }

    return cards;
}

/// Compile-time hole card parsing - returns exactly 2 cards
pub fn mustParseHoleCards(comptime card_string: []const u8) [2]Card {
    if (card_string.len != 4) {
        @compileError("Hole cards must be exactly 4 characters (e.g., 'AhAs'): " ++ card_string);
    }
    const cards = mustParseCards(card_string);
    return [2]Card{ cards[0], cards[1] };
}

/// Generate all combinations for a pocket pair (6 combinations)
pub fn generatePocketPair(rank: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 6);

    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations for pocket pairs
    for (suits, 0..) |suit1, i| {
        for (suits[i + 1 ..]) |suit2| {
            combinations[idx] = [2]Card{
                createCard(suit1, rank),
                createCard(suit2, rank),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Generate all suited combinations (4 combinations)
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 4);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]Card{
            createCard(suit, rank1),
            createCard(suit, rank2),
        };
    }

    return combinations;
}

/// Generate all offsuit combinations (12 combinations)
pub fn generateOffsuitCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 12);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations where suits are different
    for (suits) |suit1| {
        for (suits) |suit2| {
            if (suit1 != suit2) {
                combinations[idx] = [2]Card{
                    createCard(suit1, rank1),
                    createCard(suit2, rank2),
                };
                idx += 1;
            }
        }
    }

    return combinations;
}

/// Generate all combinations (suited + offsuit = 16 combinations)
pub fn generateAllCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 16);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all possible suit combinations
    for (suits) |suit1| {
        for (suits) |suit2| {
            combinations[idx] = [2]Card{
                createCard(suit1, rank1),
                createCard(suit2, rank2),
            };
            idx += 1;
        }
    }

    return combinations;
}

// Generate random hand
pub fn generateRandomHand(random: std.Random) Hand {
    var hand_bits: u64 = 0;
    var used_cards = std.StaticBitSet(52).initEmpty();

    // Generate 7 unique random cards and compute bits directly
    var cards_added: u8 = 0;
    while (cards_added < 7) {
        const card_idx = random.uintLessThan(u8, 52);
        if (!used_cards.isSet(card_idx)) {
            used_cards.set(card_idx);

            // Compute bit directly without creating Card object
            const card_bit = @as(u64, 1) << @intCast(card_idx);
            hand_bits |= card_bit;

            cards_added += 1;
        }
    }

    return Hand{ .bits = hand_bits };
}

// Sample remaining cards avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: []const Card, num_cards: u8, rng: std.Random) Hand {
    var used_bits: u64 = 0;
    for (used_cards) |card| {
        used_bits |= card.bits;
    }
    const sampled_bits = sampleRemainingCardsBits(used_bits, num_cards, rng);
    return Hand{ .bits = sampled_bits };
}

// Helper functions
inline fn cardsToBits(cards: []const Card) u64 {
    var bits: u64 = 0;
    for (cards) |card| {
        bits |= card.bits;
    }
    return bits;
}

inline fn sampleRemainingCardsBits(used_cards: u64, num_cards: u8, rng: std.Random) u64 {
    var sampled_cards: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        // Skip if card already used or sampled
        if ((used_cards & card_bit) != 0 or (sampled_cards & card_bit) != 0) {
            continue;
        }

        sampled_cards |= card_bit;
        cards_sampled += 1;
    }

    return sampled_cards;
}

// Basic tests - only essential types and conversion tests
const testing = std.testing;

test "card creation and properties" {
    const card = createCard(.spades, .ace);
    try testing.expect(card.getRank() == 14);
    try testing.expect(card.getSuit() == 1);
}

test "hand rank ordering" {
    try testing.expect(@intFromEnum(HandRank.high_card) < @intFromEnum(HandRank.pair));
    try testing.expect(@intFromEnum(HandRank.pair) < @intFromEnum(HandRank.straight_flush));
}

test "mustParseHoleCards helper" {
    const aa = mustParseHoleCards("AhAs");
    try testing.expect(aa[0].getRank() == 14 and aa[0].getSuit() == 0); // Ah
    try testing.expect(aa[1].getRank() == 14 and aa[1].getSuit() == 1); // As

    const kq = mustParseHoleCards("KdQc");
    try testing.expect(kq[0].getRank() == 13 and kq[0].getSuit() == 2); // Kd
    try testing.expect(kq[1].getRank() == 12 and kq[1].getSuit() == 3); // Qc
}

test "hand evaluation bridge to evaluator" {
    // For unit testing, we'll use a mock evaluator that always returns a fixed value
    const mockEvaluator = struct {
        fn evaluateHand(hand_bits: u64) u16 {
            _ = hand_bits;
            return 1; // Return straight flush rank
        }
    }.evaluateHand;
    
    // Test that we can create hands and evaluate them
    const royal_flush = createHand(&.{
        .{ .spades, .ace },
        .{ .spades, .king },
        .{ .spades, .queen },
        .{ .spades, .jack },
        .{ .spades, .ten },
        .{ .hearts, .two },
        .{ .hearts, .three },
    });
    
    const result = royal_flush.evaluate(mockEvaluator);
    // Should evaluate to straight flush with our mock
    try testing.expect(result == HandRank.straight_flush);
}