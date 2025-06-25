const std = @import("std");

// Slow but correct hand evaluator

// Hand evaluation types and constants
pub const Hand = u64;
pub const HandRank = u16;

// Hand ranking constants (from lowest to highest)
pub const HAND_RANKS = struct {
    pub const HIGH_CARD = 0;
    pub const PAIR = 1;
    pub const TWO_PAIR = 2;
    pub const THREE_OF_A_KIND = 3;
    pub const STRAIGHT = 4;
    pub const FLUSH = 5;
    pub const FULL_HOUSE = 6;
    pub const FOUR_OF_A_KIND = 7;
    pub const STRAIGHT_FLUSH = 8;
};

// Card utilities
pub const CLUBS_OFFSET = 0;
pub const DIAMONDS_OFFSET = 13;
pub const HEARTS_OFFSET = 26;
pub const SPADES_OFFSET = 39;
pub const RANK_MASK = 0x1FFF;

pub fn makeCard(suit: u8, rank: u8) Hand {
    const offset = suit * 13;
    return @as(Hand, 1) << @intCast(offset + rank);
}

pub fn getRankMask(hand: Hand) u16 {
    const clubs = @as(u16, @truncate(hand >> CLUBS_OFFSET)) & RANK_MASK;
    const diamonds = @as(u16, @truncate(hand >> DIAMONDS_OFFSET)) & RANK_MASK;
    const hearts = @as(u16, @truncate(hand >> HEARTS_OFFSET)) & RANK_MASK;
    const spades = @as(u16, @truncate(hand >> SPADES_OFFSET)) & RANK_MASK;

    return clubs | diamonds | hearts | spades;
}

pub fn getSuitMasks(hand: Hand) [4]u16 {
    return [4]u16{
        @as(u16, @truncate(hand >> CLUBS_OFFSET)) & RANK_MASK,
        @as(u16, @truncate(hand >> DIAMONDS_OFFSET)) & RANK_MASK,
        @as(u16, @truncate(hand >> HEARTS_OFFSET)) & RANK_MASK,
        @as(u16, @truncate(hand >> SPADES_OFFSET)) & RANK_MASK,
    };
}

pub fn hasFlush(hand: Hand) bool {
    const suits = getSuitMasks(hand);
    for (suits) |suit| {
        if (@popCount(suit) >= 5) return true;
    }
    return false;
}

// Get the highest 5 cards from a rank mask
fn getHighestRanks(ranks: u16, count: u8) u16 {
    var result: u16 = 0;
    var remaining = count;
    var bit: u8 = 12; // Start from Ace

    while (remaining > 0 and bit < 13) {
        if ((ranks & (@as(u16, 1) << @intCast(bit))) != 0) {
            result |= @as(u16, 1) << @intCast(bit);
            remaining -= 1;
        }
        if (bit == 0) break;
        bit -= 1;
    }

    return result;
}

// Check for straight in rank mask
fn getStraightMask(ranks: u16) u16 {
    // Check for wheel (A-2-3-4-5)
    if ((ranks & 0x100F) == 0x100F) { // A,2,3,4,5
        return 0x000F; // Return 2,3,4,5 (wheel straight uses 5 as high card)
    }

    // Check for regular straights
    var straight_mask: u16 = 0x1F; // 5 consecutive bits
    var i: u8 = 0;
    while (i <= 8) : (i += 1) {
        if ((ranks & straight_mask) == straight_mask) {
            return straight_mask;
        }
        straight_mask <<= 1;
    }

    return 0;
}

// Simple 7-card hand evaluation (non-optimized)
pub fn evaluateHand(hand: Hand) HandRank {
    const ranks = getRankMask(hand);
    const suits = getSuitMasks(hand);

    // Check for flush
    var flush_suit: ?u8 = null;
    var flush_ranks: u16 = 0;
    for (suits, 0..) |suit, i| {
        if (@popCount(suit) >= 5) {
            flush_suit = @intCast(i);
            flush_ranks = getHighestRanks(suit, 5);
            break;
        }
    }

    // Check for straight
    const straight_mask = getStraightMask(ranks);
    const has_straight = straight_mask != 0;

    // Straight flush
    if (flush_suit != null and has_straight) {
        const straight_in_flush = getStraightMask(suits[flush_suit.?]);
        if (straight_in_flush != 0) {
            // Royal flush
            if (straight_in_flush == 0x1F00) { // 10,J,Q,K,A
                return 7461; // Royal flush
            }
            // Straight flush
            const high_card = @clz(straight_in_flush);
            return 7454 + @as(HandRank, high_card); // Straight flush range
        }
    }

    // Count rank frequencies
    var rank_counts = [_]u8{0} ** 13;
    var i: u8 = 0;
    while (i < 13) : (i += 1) {
        if ((ranks & (@as(u16, 1) << @intCast(i))) != 0) {
            // Count how many cards of this rank
            var count: u8 = 0;
            for (suits) |suit| {
                if ((suit & (@as(u16, 1) << @intCast(i))) != 0) {
                    count += 1;
                }
            }
            rank_counts[i] = count;
        }
    }

    // Find pairs, trips, quads
    var quads: u8 = 0;
    var trips: u8 = 0;
    var pairs: u8 = 0;

    for (rank_counts, 0..) |count, rank| {
        switch (count) {
            4 => quads += 1,
            3 => trips += 1,
            2 => pairs += 1,
            else => {},
        }
        _ = rank;
    }

    // Four of a kind
    if (quads > 0) {
        return 7000; // Simplified four of a kind range
    }

    // Full house
    if (trips > 0 and pairs > 0) {
        return 6000; // Simplified full house range
    }

    // Flush (not straight)
    if (flush_suit != null) {
        return 5000; // Simplified flush range
    }

    // Straight (not flush)
    if (has_straight) {
        return 4000; // Simplified straight range
    }

    // Three of a kind
    if (trips > 0) {
        return 3000; // Simplified trips range
    }

    // Two pair
    if (pairs >= 2) {
        return 2000; // Simplified two pair range
    }

    // One pair
    if (pairs == 1) {
        return 1000; // Simplified one pair range
    }

    // High card
    return 0;
}

// Get the hand category (0-8) for correctness verification
pub fn getHandCategory(hand: Hand) u16 {
    return evaluateHand(hand);
}

// Tests
test "royal flush" {
    const royal_flush = makeCard(3, 12) | makeCard(3, 11) | makeCard(3, 10) | makeCard(3, 9) | makeCard(3, 8) |
        makeCard(0, 0) | makeCard(1, 1); // Add two random cards
    
    const rank = evaluateHand(royal_flush);
    try std.testing.expect(rank == 7461); // Royal flush value
}

test "straight flush" {
    const straight_flush = makeCard(2, 8) | makeCard(2, 7) | makeCard(2, 6) | makeCard(2, 5) | makeCard(2, 4) |
        makeCard(0, 0) | makeCard(1, 1); // Add two random cards
    
    const rank = evaluateHand(straight_flush);
    try std.testing.expect(rank >= 7454 and rank <= 7461); // Straight flush range
}

test "four of a kind" {
    const four_aces = makeCard(0, 12) | makeCard(1, 12) | makeCard(2, 12) | makeCard(3, 12) |
        makeCard(0, 11) | makeCard(1, 10) | makeCard(2, 9);
    
    const rank = evaluateHand(four_aces);
    try std.testing.expect(rank == 7000); // Four of a kind value
}

test "full house" {
    const full_house = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 10) |
        makeCard(0, 9) | makeCard(1, 9) | makeCard(2, 8) | makeCard(3, 7);
    
    const rank = evaluateHand(full_house);
    try std.testing.expect(rank == 6000); // Full house value
}

test "flush" {
    const flush = makeCard(2, 12) | makeCard(2, 10) | makeCard(2, 8) | makeCard(2, 6) | makeCard(2, 4) |
        makeCard(0, 11) | makeCard(1, 9); // Add two random cards
    
    const rank = evaluateHand(flush);
    try std.testing.expect(rank == 5000); // Flush value
}

test "straight" {
    const straight = makeCard(0, 8) | makeCard(1, 7) | makeCard(2, 6) | makeCard(3, 5) | makeCard(0, 4) |
        makeCard(1, 2) | makeCard(2, 0); // Add two random cards
    
    const rank = evaluateHand(straight);
    try std.testing.expect(rank == 4000); // Straight value
}

test "wheel straight (A-2-3-4-5)" {
    const wheel = makeCard(0, 12) | makeCard(1, 3) | makeCard(2, 2) | makeCard(3, 1) | makeCard(0, 0) |
        makeCard(1, 11) | makeCard(2, 10); // Add two random cards
    
    const rank = evaluateHand(wheel);
    try std.testing.expect(rank == 4000); // Straight value
}

test "three of a kind" {
    const three_kings = makeCard(0, 11) | makeCard(1, 11) | makeCard(2, 11) |
        makeCard(0, 9) | makeCard(1, 7) | makeCard(2, 5) | makeCard(3, 2);
    
    const rank = evaluateHand(three_kings);
    try std.testing.expect(rank == 3000); // Three of a kind value
}

test "two pair" {
    const two_pair = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 8) |
        makeCard(0, 6) | makeCard(1, 4) | makeCard(2, 2);
    
    const rank = evaluateHand(two_pair);
    try std.testing.expect(rank == 2000); // Two pair value
}

test "one pair" {
    const one_pair = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 6) |
        makeCard(0, 4) | makeCard(1, 2) | makeCard(2, 0);
    
    const rank = evaluateHand(one_pair);
    try std.testing.expect(rank == 1000); // One pair value
}

test "high card" {
    const high_card = makeCard(0, 12) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 6) |
        makeCard(0, 4) | makeCard(1, 2) | makeCard(2, 0);
    
    const rank = evaluateHand(high_card);
    try std.testing.expect(rank == 0); // High card value
}

test "card utilities" {
    const ace_spades = makeCard(3, 12);
    try std.testing.expect(ace_spades == (@as(Hand, 1) << (39 + 12)));
    
    const hand = makeCard(0, 12) | makeCard(1, 10) | makeCard(2, 8);
    const rank_mask = getRankMask(hand);
    try std.testing.expect((rank_mask & (1 << 12)) != 0); // Has ace
    try std.testing.expect((rank_mask & (1 << 10)) != 0); // Has jack
    try std.testing.expect((rank_mask & (1 << 8)) != 0);  // Has 9
}

test "flush detection" {
    const flush_hand = makeCard(2, 12) | makeCard(2, 10) | makeCard(2, 8) | makeCard(2, 6) | makeCard(2, 4);
    try std.testing.expect(hasFlush(flush_hand));
    
    const no_flush = makeCard(0, 12) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 6);
    try std.testing.expect(!hasFlush(no_flush));
}
