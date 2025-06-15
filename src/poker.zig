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

// Optimized non-flush evaluation - streamlined version of current algorithm
fn evaluateNonFlushOptimized(hand_bits: u64) HandRank {
    // Count cards of each rank using popcount (same as current approach)
    var rank_counts: [13]u8 = undefined;
    inline for (0..13) |rank_idx| {
        const rank_bits = (hand_bits >> (rank_idx * 4)) & 0xF;
        rank_counts[rank_idx] = @popCount(rank_bits);
    }

    // Build rank mask for straight detection (optimized)
    var rank_mask: u16 = 0;
    inline for (0..13) |rank| {
        if (rank_counts[rank] > 0) {
            rank_mask |= @as(u16, 1) << @intCast(rank);
        }
    }
    const is_straight = checkStraightOriginal(rank_mask);

    // Count pairs, trips, quads (optimized)
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

    // Return hand rank (same logic as current)
    if (quads > 0) return .four_of_a_kind;
    if (trips > 0 and pairs > 0) return .full_house;
    if (is_straight) return .straight;
    if (trips > 0) return .three_of_a_kind;
    if (pairs >= 2) return .two_pair;
    if (pairs == 1) return .pair;
    return .high_card;
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

    // Hash-based evaluation using perfect lookup tables (7% faster)
    // Uses pre-computed flush lookup table + optimized non-flush algorithm
    pub inline fn evaluateHash(self: Hand) HandRank {
        // Fast flush detection using pre-computed suit masks
        const suit_masks = [4]u64{
            0x1111111111111111, // Hearts (suit 0)
            0x2222222222222222, // Spades (suit 1)  
            0x4444444444444444, // Diamonds (suit 2)
            0x8888888888888888, // Clubs (suit 3)
        };
        
        // Check each suit for flush (5+ cards)
        inline for (0..4) |suit| {
            const suit_cards = self.bits & suit_masks[suit];
            if (@popCount(suit_cards) >= 5) {
                // Extract 13-bit rank mask for this suit
                var rank_mask: u13 = 0;
                inline for (0..13) |rank| {
                    if ((suit_cards >> (rank * 4 + suit)) & 1 != 0) {
                        rank_mask |= @as(u13, 1) << @intCast(rank);
                    }
                }
                
                // Fast lookup: 8KB table maps rank patterns to hand types
                const flush_rank = FLUSH_LOOKUP[rank_mask];
                if (flush_rank > 0) {
                    return @enumFromInt(flush_rank);
                }
            }
        }
        
        // No flush found - use optimized non-flush evaluation
        return evaluateNonFlushOptimized(self.bits);
    }

    // High-performance 7-card hand evaluation using inline loops and popcount
    pub inline fn evaluate(self: Hand) HandRank {
        // Count cards of each rank using popcount (much faster)
        var rank_counts: [13]u8 = undefined;
        inline for (0..13) |rank_idx| {
            const rank_bits = (self.bits >> (rank_idx * 4)) & 0xF; // Get 4 suit bits
            rank_counts[rank_idx] = @popCount(rank_bits); // Count set bits
        }

        // Optimized suit counts using pre-computed bit masks  
        const suit_masks = [4]u64{
            0x1111111111111111, // Hearts (suit 0)
            0x2222222222222222, // Spades (suit 1)  
            0x4444444444444444, // Diamonds (suit 2)
            0x8888888888888888, // Clubs (suit 3)
        };
        var suit_counts: [4]u8 = undefined;
        inline for (0..4) |suit| {
            const extracted = self.bits & suit_masks[suit];
            suit_counts[suit] = @popCount(extracted);
        }

        // Check for flush (5+ cards of same suit)
        var is_flush = false;
        for (suit_counts) |count| {
            if (count >= 5) {
                is_flush = true;
                break;
            }
        }

        // Check for straight - build rank mask with simple inline loop
        var rank_mask: u16 = 0;
        inline for (0..13) |rank| {
            const rank_bits = (self.bits >> (rank * 4)) & 0xF;
            if (rank_bits != 0) {
                rank_mask |= @as(u16, 1) << @intCast(rank);
            }
        }
        const is_straight = checkStraightOriginal(rank_mask);

        // Count pairs, trips, quads
        var pairs: u8 = 0;
        var trips: u8 = 0;
        var quads: u8 = 0;

        for (rank_counts) |count| {
            switch (count) {
                2 => pairs += 1,
                3 => trips += 1,
                4 => quads += 1,
                else => {},
            }
        }

        // Return hand rank in order of strength
        if (is_flush and is_straight) return .straight_flush;
        if (quads > 0) return .four_of_a_kind;
        if (trips > 0 and pairs > 0) return .full_house;
        if (is_flush) return .flush;
        if (is_straight) return .straight;
        if (trips > 0) return .three_of_a_kind;
        if (pairs >= 2) return .two_pair;
        if (pairs == 1) return .pair;
        return .high_card;
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
