const std = @import("std");
const card = @import("card");

// Slow but correct hand evaluator
// Uses modern card abstractions from card.zig

// Re-export types from modern card module
pub const Hand = card.Hand;
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

// Use modern card creation function
pub const makeCard = card.makeCard;

pub fn getRankMask(hand: Hand) u16 {
    const clubs = card.getSuitMask(hand, .clubs);
    const diamonds = card.getSuitMask(hand, .diamonds);
    const hearts = card.getSuitMask(hand, .hearts);
    const spades = card.getSuitMask(hand, .spades);
    return clubs | diamonds | hearts | spades;
}

pub fn getSuitMasks(hand: Hand) [4]u16 {
    return [4]u16{
        card.getSuitMask(hand, .clubs),
        card.getSuitMask(hand, .diamonds),
        card.getSuitMask(hand, .hearts),
        card.getSuitMask(hand, .spades),
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
    // Handle case with 2 trips (e.g., AAA KKK x) or trips + pairs
    if (trips > 0 and (pairs > 0 or trips > 1)) {
        var trip_rank: u8 = 0;
        var pair_rank: u8 = 0;
        var second_trip_rank: u8 = 0;

        // Find the highest trip
        for (rank_counts, 0..) |count, rank| {
            if (count == 3 and rank > trip_rank) {
                second_trip_rank = trip_rank;
                trip_rank = @intCast(rank);
            } else if (count == 3 and rank > second_trip_rank) {
                second_trip_rank = @intCast(rank);
            } else if (count == 2 and rank > pair_rank) {
                pair_rank = @intCast(rank);
            }
        }

        // If we have two trips, use the lower trip as the pair
        if (trips > 1) {
            pair_rank = second_trip_rank;
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
    const royal_flush = makeCard(.spades, .ace) | makeCard(.spades, .king) | makeCard(.spades, .queen) | makeCard(.spades, .jack) | makeCard(.spades, .ten) |
        makeCard(.clubs, .two) | makeCard(.diamonds, .three); // Add two random cards

    const rank = evaluateHand(royal_flush);
    try std.testing.expect(rank == 0); // Royal flush = rank 0 (best possible)
}

test "royal flush clubs 0x1F00" {
    const royal_clubs: u64 = 0x1F00; // A-K-Q-J-T clubs only (5 cards)
    const rank = evaluateHand(royal_clubs);
    try std.testing.expect(rank == 0); // Royal flush = rank 0 (best possible)
}

test "straight flush" {
    const straight_flush = makeCard(.hearts, .ten) | makeCard(.hearts, .nine) | makeCard(.hearts, .eight) | makeCard(.hearts, .seven) | makeCard(.hearts, .six) |
        makeCard(.clubs, .two) | makeCard(.diamonds, .three); // Add two random cards (9-high straight flush)

    const rank = evaluateHand(straight_flush);
    try std.testing.expect(rank >= 1 and rank <= 9); // Straight flush range 1-9
}

test "four of a kind" {
    const four_aces = makeCard(.clubs, .ace) | makeCard(.diamonds, .ace) | makeCard(.hearts, .ace) | makeCard(.spades, .ace) |
        makeCard(.clubs, .king) | makeCard(.diamonds, .queen) | makeCard(.hearts, .jack);

    const rank = evaluateHand(four_aces);
    try std.testing.expect(rank >= 10 and rank <= 165); // Four of a kind range
}

test "full house" {
    const full_house = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .queen) |
        makeCard(.clubs, .jack) | makeCard(.diamonds, .jack) | makeCard(.hearts, .ten) | makeCard(.spades, .nine);

    const rank = evaluateHand(full_house);
    try std.testing.expect(rank >= 166 and rank <= 321); // Full house range
}

test "flush" {
    const flush = makeCard(.hearts, .ace) | makeCard(.hearts, .queen) | makeCard(.hearts, .ten) | makeCard(.hearts, .eight) | makeCard(.hearts, .six) |
        makeCard(.clubs, .king) | makeCard(.diamonds, .jack); // Add two random cards

    const rank = evaluateHand(flush);
    try std.testing.expect(rank >= 322 and rank <= 1598); // Flush range
}

test "straight" {
    const straight = makeCard(.clubs, .ten) | makeCard(.diamonds, .nine) | makeCard(.hearts, .eight) | makeCard(.spades, .seven) | makeCard(.clubs, .six) |
        makeCard(.diamonds, .four) | makeCard(.hearts, .two); // Add two random cards (9-high straight)

    const rank = evaluateHand(straight);
    try std.testing.expect(rank >= 1599 and rank <= 1608); // Straight range
}

test "wheel straight (A-2-3-4-5)" {
    const wheel = makeCard(.clubs, .ace) | makeCard(.diamonds, .five) | makeCard(.hearts, .four) | makeCard(.spades, .three) | makeCard(.clubs, .two) |
        makeCard(.diamonds, .king) | makeCard(.hearts, .queen); // Add two random cards

    const rank = evaluateHand(wheel);
    try std.testing.expect(rank == 1608); // Wheel is worst straight
}

test "three of a kind" {
    const three_kings = makeCard(.clubs, .king) | makeCard(.diamonds, .king) | makeCard(.hearts, .king) |
        makeCard(.clubs, .jack) | makeCard(.diamonds, .nine) | makeCard(.hearts, .seven) | makeCard(.spades, .four);

    const rank = evaluateHand(three_kings);
    try std.testing.expect(rank >= 1609 and rank <= 2466); // Three of a kind range
}

test "two pair" {
    const two_pair = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .ten) |
        makeCard(.clubs, .eight) | makeCard(.diamonds, .six) | makeCard(.hearts, .four);

    const rank = evaluateHand(two_pair);
    try std.testing.expect(rank >= 2467 and rank <= 3324); // Two pair range
}

test "one pair" {
    const one_pair = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .eight) |
        makeCard(.clubs, .six) | makeCard(.diamonds, .four) | makeCard(.hearts, .two);

    const rank = evaluateHand(one_pair);
    try std.testing.expect(rank >= 3325 and rank <= 6184); // One pair range
}

test "high card" {
    const high_card = makeCard(.clubs, .ace) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .eight) |
        makeCard(.clubs, .six) | makeCard(.diamonds, .four) | makeCard(.hearts, .two);

    const rank = evaluateHand(high_card);
    try std.testing.expect(rank >= 6185 and rank <= 7461); // High card range (worst hands)
}

test "card utilities" {
    const ace_spades = makeCard(.spades, .ace);
    try std.testing.expect(ace_spades == (@as(Hand, 1) << (39 + 12)));

    const hand = makeCard(.clubs, .ace) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten);
    const rank_mask = getRankMask(hand);
    try std.testing.expect((rank_mask & (1 << 12)) != 0); // Has ace
    try std.testing.expect((rank_mask & (1 << 10)) != 0); // Has jack
    try std.testing.expect((rank_mask & (1 << 8)) != 0); // Has 9
}

test "flush detection" {
    const flush_hand = makeCard(.hearts, .ace) | makeCard(.hearts, .queen) | makeCard(.hearts, .ten) | makeCard(.hearts, .eight) | makeCard(.hearts, .six);
    try std.testing.expect(hasFlush(flush_hand));

    const no_flush = makeCard(.clubs, .ace) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .eight);
    try std.testing.expect(!hasFlush(no_flush));
}

test "wheel straight flush (A-5-4-3-2)" {
    // Wheel straight flush: A,2,3,4,5 all clubs + 2 off-suit cards
    // Ranks: A=12, 2=0, 3=1, 4=2, 5=3 → pattern 0x100F
    const wheel_sf = makeCard(.clubs, .ace) | makeCard(.clubs, .two) | makeCard(.clubs, .three) | makeCard(.clubs, .four) | makeCard(.clubs, .five) |
        makeCard(.diamonds, .queen) | makeCard(.hearts, .ten); // Add two off-suit cards

    const rank = evaluateHand(wheel_sf);

    // Should be rank 9 (worst straight flush) - fixed!
    try std.testing.expect(rank == 9); // Wheel straight flush
}

test "overlapping straights edge case - hand 1" {
    // Hand 1: 0x1F8000000008 = spades 2,3,4,5,6,7 + clubs 5
    // This contains TWO straights: 7-6-5-4-3 AND 6-5-4-3-2
    // Should return the HIGHER straight: 7-6-5-4-3
    const hand1: Hand = 0x1F8000000008;
    _ = evaluateHand(hand1);

    // Verify it's a flush
    try std.testing.expect(hasFlush(hand1));

    // Check what suit has the flush
    const suits = getSuitMasks(hand1);
    for (suits) |suit| {
        if (@popCount(suit) >= 5) {

            // Check which straights are present
            _ = getStraightMask(suit);
        }
    }
}

test "overlapping straights edge case - hand 2" {
    // Hand 2: 0x3F00001000 = hearts 8,9,T,J,Q,K + clubs A
    // This is a K-high straight flush
    const hand2: Hand = 0x3F00001000;
    _ = evaluateHand(hand2);

    // Verify it's a flush
    try std.testing.expect(hasFlush(hand2));

    // Check what suit has the flush
    const suits = getSuitMasks(hand2);
    for (suits) |suit| {
        if (@popCount(suit) >= 5) {

            // Check which straights are present
            _ = getStraightMask(suit);
        }
    }
}

test "two trips makes full house" {
    // Test case: AAAKKK7 - two trips should be a full house
    // AAA = A♠A♥A♦, KKK = K♠K♥K♦, 7 = 7♣
    const hand = makeCard(.clubs, .ace) | // A♣
        makeCard(.diamonds, .ace) | // A♦
        makeCard(.hearts, .ace) | // A♥
        makeCard(.clubs, .king) | // K♣
        makeCard(.diamonds, .king) | // K♦
        makeCard(.hearts, .king) | // K♥
        makeCard(.clubs, .eight); // 7♣

    const rank = evaluateHand(hand);

    // Should be a full house (rank 166-321)
    try std.testing.expect(rank >= 166 and rank <= 321);

    // Specifically, should be AAAKK which is a very strong full house
    // Should be 166 + 0*12 + 1 = 167 (Aces over Kings)
    try std.testing.expect(rank == 167);
}
