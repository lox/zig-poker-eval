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


// Perfect Hash Lookup Tables for 7-card evaluation
// Generated at compile time for zero runtime overhead

// Flush lookup table: 8KB table for instant flush/straight-flush detection
// Index: 13-bit mask representing which ranks are present in a suit
// Value: Hand rank (0 = not enough cards, 6 = flush, 9 = straight flush)
const FLUSH_LOOKUP = generateFlushTable();

// Rank Distribution LUT: Smaller table for instant non-flush hand categorization
// Using a simpler hash function to map rank distributions to hand categories
// Hash based on pair/trip/quad counts instead of full enumeration
const RANK_CATEGORY_LUT = generateRankCategoryLut();

// Generate flush lookup table at compile time
fn generateFlushTable() [8192]u16 {
    @setEvalBranchQuota(100000); // Increase compile-time loop limit
    var table: [8192]u16 = [_]u16{0} ** 8192;
    
    // For each possible 13-bit rank combination
    for (0..8192) |mask| {
        const popcount = @popCount(@as(u13, @intCast(mask)));
        if (popcount < 5) {
            table[mask] = 0; // Not enough cards for flush
            continue;
        }
        
        // Check for straight flush
        if (checkStraightInMask(@intCast(mask))) {
            table[mask] = 9; // Straight flush (highest rank)
        } else {
            table[mask] = 6; // Regular flush
        }
    }
    
    return table;
}

// Check if a rank mask contains a straight (for flush evaluation)
fn checkStraightInMask(mask: u13) bool {
    // Check wheel (A-2-3-4-5): bits 12,0,1,2,3
    if ((mask & 0b1000000001111) == 0b1000000001111) return true;
    
    // Check all 9 possible regular straights with unrolled loop
    const straight_patterns = [_]u13{
        0b1111100000000, // A-K-Q-J-T
        0b0111110000000, // K-Q-J-T-9
        0b0011111000000, // Q-J-T-9-8
        0b0001111100000, // J-T-9-8-7
        0b0000111110000, // T-9-8-7-6
        0b0000011111000, // 9-8-7-6-5
        0b0000001111100, // 8-7-6-5-4
        0b0000000111110, // 7-6-5-4-3
        0b0000000011111, // 6-5-4-3-2
    };
    
    inline for (straight_patterns) |pattern| {
        if ((mask & pattern) == pattern) return true;
    }
    
    return false;
}

// Simple hash function for rank category determination
// Based on pair/trip/quad counts instead of full rank distribution
fn hashRankCategory(pairs: u8, trips: u8, quads: u8) u8 {
    // Simple hash: quads*16 + trips*4 + pairs
    // Max value: 1*16 + 2*4 + 6 = 30 (well within u8 range)
    return quads * 16 + trips * 4 + pairs;
}

// Generate simplified rank category lookup table at compile time  
// Much smaller table based on pair/trip/quad combinations
fn generateRankCategoryLut() [64]HandRank {
    @setEvalBranchQuota(100000); // Increase compile-time loop limit
    var lut: [64]HandRank = [_]HandRank{.high_card} ** 64;
    
    // Enumerate all valid combinations of pairs/trips/quads for 7 cards
    for (0..2) |quads| { // 0 or 1 quad possible
        for (0..3) |trips| { // 0, 1, or 2 trips possible  
            for (0..7) |pairs| { // 0 to 6 pairs possible
                // Check if combination is valid for 7 cards
                const total_cards = quads * 4 + trips * 3 + pairs * 2;
                if (total_cards <= 7) {
                    const key = hashRankCategory(@intCast(pairs), @intCast(trips), @intCast(quads));
                    
                    // Determine hand rank from counts
                    if (quads > 0) {
                        lut[key] = .four_of_a_kind;
                    } else if (trips > 0 and pairs > 0) {
                        lut[key] = .full_house;
                    } else if (trips > 0) {
                        lut[key] = .three_of_a_kind;
                    } else if (pairs >= 2) {
                        lut[key] = .two_pair;
                    } else if (pairs == 1) {
                        lut[key] = .pair;
                    } else {
                        lut[key] = .high_card;
                    }
                }
            }
        }
    }
    
    return lut;
}

// Ultra-optimized flush detection using parallel suit extraction
// Eliminates nested loops and redundant rank mask building
inline fn detectFlushOptimized(hand_bits: u64) u16 {
    // Parallel suit count extraction using bit manipulation
    const suit_masks = [4]u64{
        0x1111111111111111, // Hearts (suit 0)
        0x2222222222222222, // Spades (suit 1)  
        0x4444444444444444, // Diamonds (suit 2)
        0x8888888888888888, // Clubs (suit 3)
    };
    
    // Check all suits in parallel for 5+ cards
    inline for (0..4) |suit| {
        const suit_cards = hand_bits & suit_masks[suit];
        const suit_count = @popCount(suit_cards);
        
        if (suit_count >= 5) {
            // Fast rank mask extraction using bit manipulation tricks
            const rank_mask = extractFlushRankMaskOptimized(suit_cards, suit);
            const flush_rank = FLUSH_LOOKUP[rank_mask];
            if (flush_rank > 0) {
                return flush_rank;
            }
        }
    }
    
    return 0; // No flush found
}

// Optimized rank mask extraction for flush suits using bit manipulation
inline fn extractFlushRankMaskOptimized(suit_cards: u64, suit: u3) u13 {
    // Use bit shifting and parallel extraction to build rank mask
    // This eliminates the 13-iteration loop in favor of bit manipulation
    var rank_mask: u13 = 0;
    
    // Parallel rank extraction - all operations can execute simultaneously on M1
    const shifted = suit_cards >> suit;
    
    // Extract all rank bits in parallel using bit manipulation
    inline for (0..13) |rank| {
        const rank_bit = (shifted >> (rank * 4)) & 1;
        rank_mask |= @as(u13, @intCast(rank_bit)) << @intCast(rank);
    }
    
    return rank_mask;
}

// Optimized rank extraction that eliminates redundancy without fighting compiler optimizations
// Extracts rank counts and builds rank mask in single pass
pub inline fn extractRankDataOptimized(hand_bits: u64) struct { counts: [13]u8, mask: u16 } {
    var rank_counts: [13]u8 = undefined;
    var rank_mask: u16 = 0;
    
    // Extract both rank counts and mask in single loop (eliminates redundancy)
    inline for (0..13) |rank_idx| {
        const rank_bits = (hand_bits >> (rank_idx * 4)) & 0xF;
        const count = @popCount(rank_bits);
        rank_counts[rank_idx] = count;
        if (count > 0) {
            rank_mask |= @as(u16, 1) << @intCast(rank_idx);
        }
    }
    
    return .{
        .counts = rank_counts,
        .mask = rank_mask,
    };
}

// Revolutionary LUT-based non-flush evaluation with straight detection optimization
// Uses 64-byte lookup table + eliminates redundant straight checks
fn evaluateNonFlushWithRankLUT(hand_bits: u64) HandRank {
    // 1. Extract rank data in single optimized pass (eliminates redundancy)
    const rank_data = extractRankDataOptimized(hand_bits);
    return evaluateNonFlushWithPrecomputedRanks(rank_data.counts, rank_data.mask);
}

// Non-flush evaluation using pre-computed rank data (eliminates redundancy)
inline fn evaluateNonFlushWithPrecomputedRanks(rank_counts: [13]u8, rank_mask: u16) HandRank {
    @setEvalBranchQuota(100000);
    // 1. Count pairs, trips, quads - optimized with unrolled loop
    var pairs: u8 = 0;
    var trips: u8 = 0;
    var quads: u8 = 0;
    
    inline for (rank_counts) |count| {
        switch (count) {
            2 => pairs += 1,
            3 => trips += 1,
            4 => quads += 1,
            else => {},
        }
    }
    
    // 2. Single lookup replaces if/else cascade
    const hash_key = hashRankCategory(pairs, trips, quads);
    const pair_category = RANK_CATEGORY_LUT[hash_key];
    
    // 3. Check for straight using pre-computed mask (no redundant work)
    const is_straight = checkStraight(rank_mask);
    
    // 4. Return best hand (HandRank enum already ordered by strength)
    return if (is_straight and @intFromEnum(HandRank.straight) > @intFromEnum(pair_category)) 
        .straight else pair_category;
}

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

// 7-card hand using bit representation for maximum performance
pub const Hand = struct {
    bits: u64, // All 7 cards as bits

    pub fn init() Hand {
        return Hand{ .bits = 0 };
    }

    pub fn addCard(self: *Hand, card: Card) void {
        self.bits |= card.bits;
    }

    pub fn fromCards(cards: [7]Card) Hand {
        var hand = init();
        for (cards) |card| {
            hand.bits |= card.bits;
        }
        return hand;
    }

    // High-performance 7-card hand evaluation with ultra-optimized flush detection
    // Uses parallel suit extraction and eliminates nested loops
    pub inline fn evaluate(self: Hand) HandRank {
        // Ultra-fast flush detection with parallel suit processing
        const flush_result = detectFlushOptimized(self.bits);
        if (flush_result > 0) {
            return @enumFromInt(flush_result);
        }
        
        // Extract rank data once for non-flush evaluation
        const rank_data = extractRankDataOptimized(self.bits);
        return evaluateNonFlushWithPrecomputedRanks(rank_data.counts, rank_data.mask);
    }
};

// Pre-computed straight patterns for lookup table optimization
const STRAIGHT_PATTERNS = [_]u16{
    0b1111100000000, // A-K-Q-J-T (royal straight)
    0b0111110000000, // K-Q-J-T-9
    0b0011111000000, // Q-J-T-9-8
    0b0001111100000, // J-T-9-8-7
    0b0000111110000, // T-9-8-7-6
    0b0000011111000, // 9-8-7-6-5
    0b0000001111100, // 8-7-6-5-4
    0b0000000111110, // 7-6-5-4-3
    0b0000000011111, // 6-5-4-3-2
    0b1000000001111, // A-5-4-3-2 (wheel)
};

// Optimized straight detection using lookup table (Priority 1 optimization)
pub inline fn checkStraight(mask: u16) bool {
    // Single loop through pre-computed patterns - should be faster than shifting
    for (STRAIGHT_PATTERNS) |pattern| {
        if ((mask & pattern) == pattern) return true;
    }
    return false;
}

// Keep original implementation for testing/comparison  
pub inline fn checkStraightOriginal(mask: u16) bool {
    // Check A-2-3-4-5 (wheel) - bits 12,0,1,2,3 (Ace is at position 12)
    if ((mask & 0b1000000001111) == 0b1000000001111) return true;

    // Check normal straights (5 consecutive bits) using shifting mask
    var check_mask: u16 = 0b11111;
    while (check_mask <= 0b1111100000000) : (check_mask <<= 1) {
        if ((mask & check_mask) == check_mask) return true;
    }

    return false;
}

// Convenience functions for creating cards with enums
pub fn createCard(suit: Suit, rank: Rank) Card {
    return Card.init(@intFromEnum(rank), @intFromEnum(suit));
}

pub fn createHand(cards: []const struct{ Suit, Rank }) Hand {
    var hand = Hand.init();
    for (cards) |card_info| {
        const card = createCard(card_info[0], card_info[1]);
        hand.addCard(card);
    }
    return hand;
}

// Parse card string like "AsKsQsJsTs2h3h" into a Hand
pub fn parseCards(card_string: []const u8) !Hand {
    if (card_string.len % 2 != 0) {
        return error.InvalidCardString;
    }
    
    var hand = Hand.init();
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
        
        hand.addCard(Card.init(rank, suit));
    }
    
    return hand;
}

// Comptime version that panics on invalid input (like Go's Must*)
pub fn mustParseCards(comptime card_string: []const u8) Hand {
    return parseCards(card_string) catch {
        @compileError("Invalid card string: " ++ card_string);
    };
}

// Tests
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

test "known hand evaluations" {
    // Royal flush
    const royal_flush = createHand(&.{
        .{ .spades, .ace },
        .{ .spades, .king },
        .{ .spades, .queen },
        .{ .spades, .jack },
        .{ .spades, .ten },
        .{ .hearts, .two },
        .{ .hearts, .three },
    });
    try testing.expect(royal_flush.evaluate() == .straight_flush);

    // Pair of aces
    const pair_hand = createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .ace },
        .{ .hearts, .two },
        .{ .spades, .four },
        .{ .diamonds, .six },
        .{ .clubs, .eight },
        .{ .hearts, .ten },
    });
    try testing.expect(pair_hand.evaluate() == .pair);

    // High card
    const high_card = createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .king },
        .{ .hearts, .queen },
        .{ .spades, .ten },
        .{ .diamonds, .eight },
        .{ .clubs, .six },
        .{ .hearts, .four },
    });
    try testing.expect(high_card.evaluate() == .high_card);
}

test "known hand patterns correctness" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const benchmark = @import("benchmark.zig");
    const torture_hands = try benchmark.generateTortureCases(allocator);
    defer allocator.free(torture_hands);

    // Verify known hand patterns evaluate correctly
    const expected_ranks = [_]HandRank{
        .straight_flush, // Royal flush
        .straight_flush, // Straight flush
        .four_of_a_kind, // Four of a kind
        .full_house,     // Full house
        .flush,          // Flush
        .straight,       // Straight
        .three_of_a_kind,// Three of a kind
        .two_pair,       // Two pair
        .pair,           // One pair
        .high_card,      // High card
        .straight,       // Wheel straight
    };

    for (torture_hands, expected_ranks) |hand, expected| {
        const result = hand.evaluate();
        try testing.expect(result == expected);
    }

    // Test random hands for basic validity
    const random_hands = try benchmark.generateRandomHands(allocator, 100, 123);
    defer allocator.free(random_hands);

    for (random_hands) |hand| {
        const result = hand.evaluate();
        try testing.expect(@intFromEnum(result) >= 1);
        try testing.expect(@intFromEnum(result) <= 9);
    }
}

test "edge cases and corner cases" {
    // Test all straight variations
    const ace_high_straight = try parseCards("AsKsQdJcTh2h3d");
    try testing.expect(ace_high_straight.evaluate() == .straight);
    
    const wheel_straight = try parseCards("Ah2s3d4c5h6s7d");
    try testing.expect(wheel_straight.evaluate() == .straight);
    
    const middle_straight = try parseCards("6h7s8d9cTh2s3d");
    try testing.expect(middle_straight.evaluate() == .straight);
    
    // Test flush vs straight priority
    const flush_beats_straight = try parseCards("AhKhQhJhTh2s3d");
    try testing.expect(flush_beats_straight.evaluate() == .straight_flush);
    
    // Test full house variations
    const trips_over_pair = try parseCards("AhAsAdKhKs2c3d");
    try testing.expect(trips_over_pair.evaluate() == .full_house);
    
    const pair_over_trips = try parseCards("AhAsKdKhKs2c3d");
    try testing.expect(pair_over_trips.evaluate() == .full_house);
    
    // Test quad variations
    const quads_with_trips = try parseCards("AhAsAdAcKhKsKd");
    try testing.expect(quads_with_trips.evaluate() == .four_of_a_kind);
    
    // Test two pair edge cases
    const high_two_pair = try parseCards("AhAsKdKh2s3c4d");
    try testing.expect(high_two_pair.evaluate() == .two_pair);
    
    const low_two_pair = try parseCards("3h3s2d2hAs5c6d");
    try testing.expect(low_two_pair.evaluate() == .two_pair);
    
    // Test minimum hands
    const ace_high = try parseCards("AhKsQdJc9h7s2d");
    try testing.expect(ace_high.evaluate() == .high_card);
    
    const deuce_high = try parseCards("2h3s4d5c7h8s9d");
    try testing.expect(deuce_high.evaluate() == .high_card);
}
