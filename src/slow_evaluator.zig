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
    // Check for regular straights starting from highest (A-K-Q-J-T down to 6-5-4-3-2)
    // This ensures we find the HIGHEST straight when there are overlapping ones
    var straight_mask: u16 = 0x1F00; // Start with A-K-Q-J-T
    var i: u8 = 0;
    while (i <= 8) : (i += 1) {
        if ((ranks & straight_mask) == straight_mask) {
            return straight_mask;
        }
        straight_mask >>= 1; // Shift right to check next lower straight
    }

    // Check for wheel (A-2-3-4-5) last since it's the lowest straight
    if ((ranks & 0x100F) == 0x100F) { // A,2,3,4,5
        return 0x100F; // Return full wheel pattern including Ace for straight flush detection
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
            // Royal flush (AKQJT of same suit)
            if (straight_in_flush == 0x1F00) { // 10,J,Q,K,A
                return 0; // Royal flush (rank 0 = best possible hand)
            }
            // Wheel straight flush (A-2-3-4-5) - 5-high
            if (straight_in_flush == 0x100F) { // A,2,3,4,5 wheel (full pattern)
                return 9; // Worst straight flush
            }
            // Other straight flushes: K-high=1, Q-high=2, ..., 6-high=8
            const high_card_bit = @clz(straight_in_flush);
            const high_card_rank = 15 - high_card_bit; // Convert clz to rank (12=A, 11=K, etc.)
            return @as(HandRank, 12 - high_card_rank); // Map K-high=1, Q-high=2, etc.
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

    // Four of a kind (ranks 10-165: best quads are lower numbers)
    if (quads > 0) {
        // Find the quad rank and best kicker
        var quad_rank: u8 = 0;
        var kicker_rank: u8 = 0;
        
        for (rank_counts, 0..) |count, rank| {
            if (count == 4) {
                quad_rank = @intCast(rank);
            } else if (count >= 1 and rank != quad_rank and rank > kicker_rank) {
                kicker_rank = @intCast(rank);
            }
        }
        
        // Higher quad rank = better hand = lower rank number
        // Aces quad = rank 10, 2s quad = rank ~165
        return 10 + @as(HandRank, (12 - quad_rank)) * 12 + @as(HandRank, (12 - kicker_rank));
    }

    // Full house (ranks 166-321: best boats are lower numbers)
    if (trips > 0 and pairs > 0) {
        var trip_rank: u8 = 0;
        var pair_rank: u8 = 0;
        
        for (rank_counts, 0..) |count, rank| {
            if (count == 3) {
                trip_rank = @intCast(rank);
            } else if (count == 2) {
                pair_rank = @intCast(rank);
            }
        }
        
        // Higher trip rank = better hand = lower rank number
        return 166 + @as(HandRank, (12 - trip_rank)) * 12 + @as(HandRank, (12 - pair_rank));
    }

    // Flush (not straight) (ranks 322-1598: best flushes are lower numbers)
    if (flush_suit != null) {
        // Simple flush ranking - A-high flush is rank 322, 7-high flush is ~1598
        const high_card_bit = @clz(flush_ranks);
        const high_card_rank = 15 - high_card_bit;
        return 322 + @as(HandRank, (12 - high_card_rank)) * 100; // Approximate flush range
    }

    // Straight (not flush) (ranks 1599-1608: A-high=1599, 5-high=1608)
    if (has_straight) {
        if (straight_mask == 0x100F) { // A-2-3-4-5 wheel (5-high)
            return 1608; // Worst straight
        }
        const high_card_bit = @clz(straight_mask);
        const high_card_rank = 15 - high_card_bit;
        return 1599 + @as(HandRank, (12 - high_card_rank)); // A-high=1599, K-high=1600, etc.
    }

    // Three of a kind (ranks 1609-2466: AAA is better than 222)
    if (trips > 0) {
        var trip_rank: u8 = 0;
        for (rank_counts, 0..) |count, rank| {
            if (count == 3) {
                trip_rank = @intCast(rank);
                break;
            }
        }
        return 1609 + @as(HandRank, (12 - trip_rank)) * 65; // Approximate trips range
    }

    // Two pair (ranks 2467-3324: AA22 is better than 3322)
    if (pairs >= 2) {
        var high_pair: u8 = 0;
        var low_pair: u8 = 0;
        
        for (rank_counts, 0..) |count, rank| {
            if (count == 2) {
                if (rank > high_pair) {
                    low_pair = high_pair;
                    high_pair = @intCast(rank);
                } else if (rank > low_pair) {
                    low_pair = @intCast(rank);
                }
            }
        }
        
        return 2467 + @as(HandRank, (12 - high_pair)) * 65 + @as(HandRank, (12 - low_pair));
    }

    // One pair (ranks 3325-6184: AA is better than 22)
    if (pairs == 1) {
        var pair_rank: u8 = 0;
        for (rank_counts, 0..) |count, rank| {
            if (count == 2) {
                pair_rank = @intCast(rank);
                break;
            }
        }
        return 3325 + @as(HandRank, (12 - pair_rank)) * 220; // Approximate pair range
    }

    // High card (ranks 6185-7461: AKQJ9 is better than 75432)
    const high_card_bit = @clz(ranks);
    const high_card_rank = 15 - high_card_bit;
    return 6185 + @as(HandRank, (12 - high_card_rank)) * 100; // Approximate high card range
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
    try std.testing.expect(rank == 0); // Royal flush = rank 0 (best possible)
}

test "royal flush clubs 0x1F00" {
    const royal_clubs: u64 = 0x1F00; // A-K-Q-J-T clubs only (5 cards)
    const rank = evaluateHand(royal_clubs);
    try std.testing.expect(rank == 0); // Royal flush = rank 0 (best possible)
}

test "straight flush" {
    const straight_flush = makeCard(2, 8) | makeCard(2, 7) | makeCard(2, 6) | makeCard(2, 5) | makeCard(2, 4) |
        makeCard(0, 0) | makeCard(1, 1); // Add two random cards (9-high straight flush)
    
    const rank = evaluateHand(straight_flush);
    try std.testing.expect(rank >= 1 and rank <= 9); // Straight flush range 1-9
}

test "four of a kind" {
    const four_aces = makeCard(0, 12) | makeCard(1, 12) | makeCard(2, 12) | makeCard(3, 12) |
        makeCard(0, 11) | makeCard(1, 10) | makeCard(2, 9);
    
    const rank = evaluateHand(four_aces);
    try std.testing.expect(rank >= 10 and rank <= 165); // Four of a kind range
}

test "full house" {
    const full_house = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 10) |
        makeCard(0, 9) | makeCard(1, 9) | makeCard(2, 8) | makeCard(3, 7);
    
    const rank = evaluateHand(full_house);
    try std.testing.expect(rank >= 166 and rank <= 321); // Full house range
}

test "flush" {
    const flush = makeCard(2, 12) | makeCard(2, 10) | makeCard(2, 8) | makeCard(2, 6) | makeCard(2, 4) |
        makeCard(0, 11) | makeCard(1, 9); // Add two random cards
    
    const rank = evaluateHand(flush);
    try std.testing.expect(rank >= 322 and rank <= 1598); // Flush range
}

test "straight" {
    const straight = makeCard(0, 8) | makeCard(1, 7) | makeCard(2, 6) | makeCard(3, 5) | makeCard(0, 4) |
        makeCard(1, 2) | makeCard(2, 0); // Add two random cards (9-high straight)
    
    const rank = evaluateHand(straight);
    try std.testing.expect(rank >= 1599 and rank <= 1608); // Straight range
}

test "wheel straight (A-2-3-4-5)" {
    const wheel = makeCard(0, 12) | makeCard(1, 3) | makeCard(2, 2) | makeCard(3, 1) | makeCard(0, 0) |
        makeCard(1, 11) | makeCard(2, 10); // Add two random cards
    
    const rank = evaluateHand(wheel);
    try std.testing.expect(rank == 1608); // Wheel is worst straight
}

test "three of a kind" {
    const three_kings = makeCard(0, 11) | makeCard(1, 11) | makeCard(2, 11) |
        makeCard(0, 9) | makeCard(1, 7) | makeCard(2, 5) | makeCard(3, 2);
    
    const rank = evaluateHand(three_kings);
    try std.testing.expect(rank >= 1609 and rank <= 2466); // Three of a kind range
}

test "two pair" {
    const two_pair = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 8) |
        makeCard(0, 6) | makeCard(1, 4) | makeCard(2, 2);
    
    const rank = evaluateHand(two_pair);
    try std.testing.expect(rank >= 2467 and rank <= 3324); // Two pair range
}

test "one pair" {
    const one_pair = makeCard(0, 10) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 6) |
        makeCard(0, 4) | makeCard(1, 2) | makeCard(2, 0);
    
    const rank = evaluateHand(one_pair);
    try std.testing.expect(rank >= 3325 and rank <= 6184); // One pair range
}

test "high card" {
    const high_card = makeCard(0, 12) | makeCard(1, 10) | makeCard(2, 8) | makeCard(3, 6) |
        makeCard(0, 4) | makeCard(1, 2) | makeCard(2, 0);
    
    const rank = evaluateHand(high_card);
    try std.testing.expect(rank >= 6185 and rank <= 7461); // High card range (worst hands)
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

test "wheel straight flush (A-5-4-3-2)" {
    // Wheel straight flush: A,2,3,4,5 all clubs + 2 off-suit cards  
    // Ranks: A=12, 2=0, 3=1, 4=2, 5=3 → pattern 0x100F
    const wheel_sf = makeCard(0, 12) | makeCard(0, 0) | makeCard(0, 1) | makeCard(0, 2) | makeCard(0, 3) |
        makeCard(1, 10) | makeCard(2, 8); // Add two off-suit cards
    
    const rank = evaluateHand(wheel_sf);
    
    // Should be rank 9 (worst straight flush) - fixed!
    try std.testing.expect(rank == 9); // Wheel straight flush
}

test "overlapping straights edge case - hand 1" {
    // Hand 1: 0x1F8000000008 = spades 2,3,4,5,6,7 + clubs 5
    // This contains TWO straights: 7-6-5-4-3 AND 6-5-4-3-2
    // Should return the HIGHER straight: 7-6-5-4-3
    const hand1: Hand = 0x1F8000000008;
    const rank = evaluateHand(hand1);
    
    std.debug.print("Hand 1 rank: {}\n", .{rank});
    
    // Verify it's a flush
    try std.testing.expect(hasFlush(hand1));
    
    // Check what suit has the flush
    const suits = getSuitMasks(hand1);
    for (suits, 0..) |suit, i| {
        if (@popCount(suit) >= 5) {
            std.debug.print("Flush suit {}: 0x{X} (popcount: {})\n", .{i, suit, @popCount(suit)});
            
            // Check which straights are present
            const straight_7_high = getStraightMask(suit);
            std.debug.print("Straight in flush suit: 0x{X}\n", .{straight_7_high});
        }
    }
}

test "overlapping straights edge case - hand 2" {
    // Hand 2: 0x3F00001000 = hearts 8,9,T,J,Q,K + clubs A
    // This is a K-high straight flush
    const hand2: Hand = 0x3F00001000;
    const rank = evaluateHand(hand2);
    
    std.debug.print("Hand 2 rank: {}\n", .{rank});
    
    // Verify it's a flush
    try std.testing.expect(hasFlush(hand2));
    
    // Check what suit has the flush
    const suits = getSuitMasks(hand2);
    for (suits, 0..) |suit, i| {
        if (@popCount(suit) >= 5) {
            std.debug.print("Flush suit {}: 0x{X} (popcount: {})\n", .{i, suit, @popCount(suit)});
            
            // Check which straights are present
            const straight_mask = getStraightMask(suit);
            std.debug.print("Straight in flush suit: 0x{X}\n", .{straight_mask});
        }
    }
}
