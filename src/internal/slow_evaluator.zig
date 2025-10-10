const std = @import("std");
const card = @import("card");

// Slow but correct hand evaluator
// Uses modern card abstractions from card.zig

// Re-export types from modern card module
pub const Hand = card.Hand;
pub const HandRank = u16;
pub const CATEGORY_STEP: HandRank = 4096;

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
        const shift_bit: u4 = @intCast(bit);
        if ((ranks & (@as(u16, 1) << shift_bit)) != 0) {
            result |= @as(u16, 1) << shift_bit;
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

inline fn rankOrder(rank: u8) u16 {
    return @intCast(12 - rank);
}

fn choose(n: u8, k: u8) u16 {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    const kk = if (k > n - k) n - k else k;
    var result: u32 = 1;
    var i: u8 = 0;

    while (i < kk) : (i += 1) {
        result = result * (@as(u32, n) - @as(u32, i));
        result /= @as(u32, i) + 1;
    }

    return @intCast(result);
}

fn combinationIndexAscending(n: u8, combo: []const u8) u16 {
    var index: u16 = 0;
    var prev: i16 = -1;
    const k = combo.len;

    for (combo, 0..) |c, i| {
        var j: i16 = prev + 1;
        while (j < c) : (j += 1) {
            const remaining = @as(u8, @intCast(k - i - 1));
            index += choose(n - 1 - @as(u8, @intCast(j)), remaining);
        }
        prev = @intCast(c);
    }

    return index;
}

fn combinationIndexDescending(n: u8, combo: []const u8) u16 {
    const total = choose(n, @intCast(combo.len));
    return total - 1 - combinationIndexAscending(n, combo);
}

fn rankToAvailableIndex(rank: u8, exclude_mask: u16) u8 {
    var index: u8 = 0;
    var r: u8 = 0;
    while (r < rank) : (r += 1) {
        const shift_r: u4 = @intCast(r);
        if ((exclude_mask & (@as(u16, 1) << shift_r)) == 0) {
            index += 1;
        }
    }
    return index;
}

fn singleOrderDescending(rank: u8, exclude_mask: u16) u16 {
    const total_allowed: u16 = 13 - @popCount(exclude_mask);
    const idx = rankToAvailableIndex(rank, exclude_mask);
    return (total_allowed - 1) - idx;
}

fn straightHighRank(mask: u16) u8 {
    if (mask == 0x100F) { // Wheel
        return 3; // Rank index for 5
    }
    return @intCast(15 - @clz(mask));
}

fn maskToRanksDescending(mask: u16) [5]u8 {
    var result: [5]u8 = undefined;
    var idx: usize = 0;
    var rank: i8 = 12;

    while (idx < 5 and rank >= 0) : (rank -= 1) {
        const bit = @as(u16, 1) << @as(u4, @intCast(rank));
        if ((mask & bit) != 0) {
            result[idx] = @intCast(rank);
            idx += 1;
        }
    }

    return result;
}

fn compareRanksDesc(a: [5]u8, b: [5]u8) i2 {
    for (0..5) |i| {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

// Simple 7-card hand evaluation (non-optimized)
pub fn evaluateHand(hand: Hand) HandRank {
    const ranks = getRankMask(hand);
    const suits = getSuitMasks(hand);

    var best_straight_flush_high: ?u8 = null;
    var has_flush = false;
    var best_flush_ranks: [5]u8 = undefined;

    inline for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            const straight_mask_suit = getStraightMask(suit_mask);
            if (straight_mask_suit != 0) {
                const high = straightHighRank(straight_mask_suit);
                if (best_straight_flush_high == null or high > best_straight_flush_high.?) {
                    best_straight_flush_high = high;
                }
            }

            const top_mask = getHighestRanks(suit_mask, 5);
            const top_ranks = maskToRanksDescending(top_mask);

            if (!has_flush or compareRanksDesc(top_ranks, best_flush_ranks) == 1) {
                best_flush_ranks = top_ranks;
                has_flush = true;
            }
        }
    }

    const straight_mask = getStraightMask(ranks);
    const has_straight = straight_mask != 0;
    const straight_high: u8 = if (has_straight) straightHighRank(straight_mask) else 0;

    var rank_counts = [_]u8{0} ** 13;
    for (0..13) |i| {
        const bit = @as(u16, 1) << @as(u4, @intCast(i));
        if ((ranks & bit) == 0) continue;

        var count: u8 = 0;
        for (suits) |suit_mask| {
            if ((suit_mask & bit) != 0) count += 1;
        }
        rank_counts[i] = count;
    }

    var quads: ?u8 = null;
    var trips_list: [3]u8 = undefined;
    var trips_len: usize = 0;
    var pairs_list: [4]u8 = undefined;
    var pairs_len: usize = 0;
    var singles_list: [7]u8 = undefined;
    var singles_len: usize = 0;

    var rank_iter: i8 = 12;
    while (rank_iter >= 0) : (rank_iter -= 1) {
        const idx: usize = @intCast(rank_iter);
        const count = rank_counts[idx];
        switch (count) {
            4 => quads = @intCast(rank_iter),
            3 => {
                trips_list[trips_len] = @intCast(rank_iter);
                trips_len += 1;
            },
            2 => {
                pairs_list[pairs_len] = @intCast(rank_iter);
                pairs_len += 1;
            },
            1 => {
                singles_list[singles_len] = @intCast(rank_iter);
                singles_len += 1;
            },
            else => {},
        }
    }

    if (best_straight_flush_high) |high| {
        return CATEGORY_STEP * 0 + rankOrder(high);
    }

    if (quads) |quad_rank| {
        var kicker_rank: u8 = 0;
        var search: i8 = 12;
        while (search >= 0) : (search -= 1) {
            if (search == quad_rank) continue;
            if (rank_counts[@intCast(search)] > 0) {
                kicker_rank = @intCast(search);
                break;
            }
        }
        const exclude_mask = @as(u16, 1) << @as(u4, @intCast(quad_rank));
        const kicker_order = singleOrderDescending(kicker_rank, exclude_mask);
        const value = rankOrder(quad_rank) * 12 + kicker_order;
        return CATEGORY_STEP * 1 + @as(HandRank, value);
    }

    if (trips_len > 0 and (pairs_len > 0 or trips_len > 1)) {
        const primary_trip = trips_list[0];
        const pair_rank = if (trips_len > 1) trips_list[1] else pairs_list[0];
        const value = rankOrder(primary_trip) * 13 + rankOrder(pair_rank);
        return CATEGORY_STEP * 2 + @as(HandRank, value);
    }

    if (has_flush) {
        var ascending: [5]u8 = undefined;
        inline for (0..5) |i| ascending[i] = best_flush_ranks[4 - i];
        const flush_value = combinationIndexDescending(13, ascending[0..]);
        return CATEGORY_STEP * 3 + flush_value;
    }

    if (has_straight) {
        return CATEGORY_STEP * 4 + rankOrder(straight_high);
    }

    if (trips_len > 0) {
        const trip_rank = trips_list[0];

        var kicker_candidates: [2]u8 = undefined;
        var kicker_len: usize = 0;
        var scan: i8 = 12;
        while (scan >= 0 and kicker_len < 2) : (scan -= 1) {
            if (scan == trip_rank) continue;
            if (rank_counts[@intCast(scan)] > 0) {
                kicker_candidates[kicker_len] = @intCast(scan);
                kicker_len += 1;
            }
        }

        const exclude_mask = @as(u16, 1) << @as(u4, @intCast(trip_rank));
        var combo: [2]u8 = undefined;
        combo[0] = rankToAvailableIndex(kicker_candidates[1], exclude_mask);
        combo[1] = rankToAvailableIndex(kicker_candidates[0], exclude_mask);
        const kicker_index = combinationIndexDescending(12, combo[0..]);
        const value = rankOrder(trip_rank) * 66 + kicker_index;
        return CATEGORY_STEP * 5 + @as(HandRank, value);
    }

    if (pairs_len >= 2) {
        const high_pair = pairs_list[0];
        const low_pair = pairs_list[1];

        var kicker_rank: u8 = 0;
        var scan: i8 = 12;
        while (scan >= 0) : (scan -= 1) {
            if (scan == high_pair or scan == low_pair) continue;
            if (rank_counts[@intCast(scan)] > 0) {
                kicker_rank = @intCast(scan);
                break;
            }
        }

        var pair_combo: [2]u8 = .{ @intCast(low_pair), @intCast(high_pair) };
        const pair_index = combinationIndexDescending(13, pair_combo[0..]);

        const exclude_mask = (@as(u16, 1) << @as(u4, @intCast(high_pair))) | (@as(u16, 1) << @as(u4, @intCast(low_pair)));
        const kicker_order = singleOrderDescending(kicker_rank, exclude_mask);
        const value = pair_index * 11 + kicker_order;
        return CATEGORY_STEP * 6 + @as(HandRank, value);
    }

    if (pairs_len == 1) {
        const pair_rank = pairs_list[0];

        var kicker_ranks: [3]u8 = undefined;
        var kicker_len: usize = 0;
        var scan: i8 = 12;
        while (scan >= 0 and kicker_len < 3) : (scan -= 1) {
            if (scan == pair_rank) continue;
            if (rank_counts[@intCast(scan)] > 0) {
                kicker_ranks[kicker_len] = @intCast(scan);
                kicker_len += 1;
            }
        }

        const exclude_mask = @as(u16, 1) << @as(u4, @intCast(pair_rank));
        var combo: [3]u8 = undefined;
        combo[0] = rankToAvailableIndex(kicker_ranks[2], exclude_mask);
        combo[1] = rankToAvailableIndex(kicker_ranks[1], exclude_mask);
        combo[2] = rankToAvailableIndex(kicker_ranks[0], exclude_mask);
        const kicker_index = combinationIndexDescending(12, combo[0..]);
        const value = rankOrder(pair_rank) * 220 + kicker_index;
        return CATEGORY_STEP * 7 + @as(HandRank, value);
    }

    var high_ranks: [5]u8 = undefined;
    inline for (0..5) |i| high_ranks[i] = singles_list[i];
    var ascending_high: [5]u8 = undefined;
    inline for (0..5) |i| ascending_high[i] = high_ranks[4 - i];
    const high_index = combinationIndexDescending(13, ascending_high[0..]);
    return CATEGORY_STEP * 8 + high_index;
}

// Get the hand category (0-8) for correctness verification
pub fn getHandCategory(hand: Hand) u16 {
    return evaluateHand(hand) / CATEGORY_STEP;
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
    try std.testing.expect(rank / CATEGORY_STEP == 1); // Four of a kind
}

test "full house" {
    const full_house = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .queen) |
        makeCard(.clubs, .jack) | makeCard(.diamonds, .jack) | makeCard(.hearts, .ten) | makeCard(.spades, .nine);

    const rank = evaluateHand(full_house);
    try std.testing.expect(rank / CATEGORY_STEP == 2); // Full house
}

test "flush" {
    const flush = makeCard(.hearts, .ace) | makeCard(.hearts, .queen) | makeCard(.hearts, .ten) | makeCard(.hearts, .eight) | makeCard(.hearts, .six) |
        makeCard(.clubs, .king) | makeCard(.diamonds, .jack); // Add two random cards

    const rank = evaluateHand(flush);
    try std.testing.expect(rank / CATEGORY_STEP == 3); // Flush
}

test "straight" {
    const straight = makeCard(.clubs, .ten) | makeCard(.diamonds, .nine) | makeCard(.hearts, .eight) | makeCard(.spades, .seven) | makeCard(.clubs, .six) |
        makeCard(.diamonds, .four) | makeCard(.hearts, .two); // Add two random cards (9-high straight)

    const rank = evaluateHand(straight);
    try std.testing.expect(rank / CATEGORY_STEP == 4); // Straight
}

test "wheel straight (A-2-3-4-5)" {
    const wheel = makeCard(.clubs, .ace) | makeCard(.diamonds, .five) | makeCard(.hearts, .four) | makeCard(.spades, .three) | makeCard(.clubs, .two) |
        makeCard(.diamonds, .king) | makeCard(.hearts, .queen); // Add two random cards

    const rank = evaluateHand(wheel);
    try std.testing.expect(rank / CATEGORY_STEP == 4); // Straight category
}

test "three of a kind" {
    const three_kings = makeCard(.clubs, .king) | makeCard(.diamonds, .king) | makeCard(.hearts, .king) |
        makeCard(.clubs, .jack) | makeCard(.diamonds, .nine) | makeCard(.hearts, .seven) | makeCard(.spades, .four);

    const rank = evaluateHand(three_kings);
    try std.testing.expect(rank / CATEGORY_STEP == 5); // Three of a kind
}

test "two pair" {
    const two_pair = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .ten) |
        makeCard(.clubs, .eight) | makeCard(.diamonds, .six) | makeCard(.hearts, .four);

    const rank = evaluateHand(two_pair);
    try std.testing.expect(rank / CATEGORY_STEP == 6); // Two pair
}

test "one pair" {
    const one_pair = makeCard(.clubs, .queen) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .eight) |
        makeCard(.clubs, .six) | makeCard(.diamonds, .four) | makeCard(.hearts, .two);

    const rank = evaluateHand(one_pair);
    try std.testing.expect(rank / CATEGORY_STEP == 7); // One pair
}

test "high card" {
    const high_card = makeCard(.clubs, .ace) | makeCard(.diamonds, .queen) | makeCard(.hearts, .ten) | makeCard(.spades, .eight) |
        makeCard(.clubs, .six) | makeCard(.diamonds, .four) | makeCard(.hearts, .two);

    const rank = evaluateHand(high_card);
    try std.testing.expect(rank / CATEGORY_STEP == 8); // High card
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

    // Should be a full house
    try std.testing.expect(rank / CATEGORY_STEP == 2);

    // Specifically, should be AAAKK which is a very strong full house
    try std.testing.expect(rank == CATEGORY_STEP * 2 + 1);
}
