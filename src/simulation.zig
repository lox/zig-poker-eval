const std = @import("std");
const poker = @import("poker.zig");

pub const ShowdownResult = struct {
    winners: []u8,
    winning_rank: poker.HandRank,

    pub fn init() ShowdownResult {
        return ShowdownResult{
            .winners = &.{},
            .winning_rank = .high_card,
        };
    }

    pub fn deinit(self: ShowdownResult, allocator: std.mem.Allocator) void {
        allocator.free(self.winners);
    }
};

const MAX_PLAYERS = 10; // Standard poker table limit

// Core showdown evaluation - determines winners from multiple hands (optimized)
pub fn evaluateShowdown(hands: []const poker.Hand, allocator: std.mem.Allocator) !ShowdownResult {
    if (hands.len == 0) return error.NoHands;
    if (hands.len > MAX_PLAYERS) return error.TooManyPlayers;

    // Use stack allocation for ranks (no heap allocation!)
    var ranks: [MAX_PLAYERS]poker.HandRank = undefined;
    var best_rank: poker.HandRank = .high_card;

    // Evaluate all hands and find best rank in single pass
    for (hands, 0..) |hand, i| {
        ranks[i] = hand.evaluate();
        if (@intFromEnum(ranks[i]) > @intFromEnum(best_rank)) {
            best_rank = ranks[i];
        }
    }

    // Count and collect winners in single pass
    var winner_count: usize = 0;
    var winner_indices: [MAX_PLAYERS]u8 = undefined;

    for (ranks[0..hands.len], 0..) |rank, i| {
        if (rank == best_rank) {
            winner_indices[winner_count] = @intCast(i);
            winner_count += 1;
        }
    }

    // Only allocate for final result
    const winners = try allocator.dupe(u8, winner_indices[0..winner_count]);

    return ShowdownResult{
        .winners = winners,
        .winning_rank = best_rank,
    };
}

// Fast path for 2-player showdown (no allocation for common case)
pub fn evaluateShowdownHeadToHead(hand1: poker.Hand, hand2: poker.Hand) struct { winner: u8, tie: bool, winning_rank: poker.HandRank } {
    const rank1 = hand1.evaluate();
    const rank2 = hand2.evaluate();

    const rank1_value = @intFromEnum(rank1);
    const rank2_value = @intFromEnum(rank2);

    if (rank1_value > rank2_value) {
        return .{ .winner = 0, .tie = false, .winning_rank = rank1 };
    } else if (rank2_value > rank1_value) {
        return .{ .winner = 1, .tie = false, .winning_rank = rank2 };
    } else {
        return .{ .winner = 0, .tie = true, .winning_rank = rank1 }; // Tie, winner doesn't matter
    }
}

// Sample remaining cards, avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: []const poker.Card, num_cards: u8, rng: std.Random, allocator: std.mem.Allocator) ![]poker.Card {
    const used_hand = poker.cardsToHand(used_cards);
    var sampled_cards: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        // Skip if card already used or sampled
        if ((used_hand.bits & card_bit) != 0 or (sampled_cards & card_bit) != 0) {
            continue;
        }

        sampled_cards |= card_bit;
        cards_sampled += 1;
    }

    // Convert sampled bits back to cards
    var result = std.ArrayList(poker.Card).init(allocator);
    defer result.deinit();

    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((sampled_cards & card_bit) != 0) {
            const rank: u8 = @intCast((i / 4) + 2);
            const suit: u2 = @intCast(i % 4);
            try result.append(poker.Card.init(rank, suit));
        }
    }

    return result.toOwnedSlice();
}

// Internal helper - sample remaining cards using bit representation for performance
// Note: public for profiling only, prefer clean wrapper functions for normal use
pub fn sampleRemainingCardsBits(used_cards: u64, num_cards: u8, rng: std.Random) u64 {
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

// Public wrapper for performance-critical equity calculations
pub fn sampleRemainingCardsForEquity(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, num_cards: u8, rng: std.Random) u64 {
    const hero_bits = hero_hole[0].bits | hero_hole[1].bits;
    const villain_bits = villain_hole[0].bits | villain_hole[1].bits;
    const board_hand = poker.cardsToHand(board);
    const used_cards = hero_bits | villain_bits | board_hand.bits;

    return sampleRemainingCardsBits(used_cards, num_cards, rng);
}

// Internal helper - combine card bits into Hand for performance
// Note: public for profiling only, prefer clean wrapper functions for normal use
pub fn combineCardsBits(hole_bits: u64, board_bits: u64) poker.Hand {
    return poker.Hand{ .bits = hole_bits | board_bits };
}

// Public wrapper for performance-critical equity calculations
pub fn combineCardsForEquity(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board_bits: u64) struct { hero: poker.Hand, villain: poker.Hand } {
    const hero_bits = hero_hole[0].bits | hero_hole[1].bits;
    const villain_bits = villain_hole[0].bits | villain_hole[1].bits;

    return .{
        .hero = combineCardsBits(hero_bits, board_bits),
        .villain = combineCardsBits(villain_bits, board_bits),
    };
}

// Internal helper - enumerate card combinations using bit representation for performance
fn enumerateCardCombinationsBits(used_cards: u64, num_cards: u8, allocator: std.mem.Allocator) ![]u64 {
    var cards = std.ArrayList(poker.Card).init(allocator);
    defer cards.deinit();

    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((used_cards & card_bit) == 0) {
            const rank: u8 = @intCast((i / 4) + 2);
            const suit: u2 = @intCast(i % 4);
            try cards.append(poker.Card.init(rank, suit));
        }
    }

    const available_cards = try cards.toOwnedSlice();
    defer allocator.free(available_cards);

    if (num_cards > available_cards.len) {
        return error.NotEnoughCards;
    }

    const total_combinations = binomial(available_cards.len, num_cards);
    var combinations = try allocator.alloc(u64, total_combinations);
    var combination_idx: usize = 0;

    // Generate all combinations using recursion
    try generateCombinationsBits(available_cards, num_cards, 0, 0, &combinations, &combination_idx);

    return combinations;
}

// Public wrapper for performance-critical equity calculations
pub fn enumerateCardCombinationsForEquity(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, num_cards: u8, allocator: std.mem.Allocator) ![]u64 {
    const hero_bits = hero_hole[0].bits | hero_hole[1].bits;
    const villain_bits = villain_hole[0].bits | villain_hole[1].bits;
    const board_hand = poker.cardsToHand(board);
    const used_cards = hero_bits | villain_bits | board_hand.bits;

    return enumerateCardCombinationsBits(used_cards, num_cards, allocator);
}

// Public wrapper for multiway equity calculations
pub fn sampleRemainingCardsForMultiway(hands: [][2]poker.Card, board: []const poker.Card, num_cards: u8, rng: std.Random) u64 {
    const board_hand = poker.cardsToHand(board);
    var used_cards: u64 = board_hand.bits;

    for (hands) |hole| {
        const hole_bits = hole[0].bits | hole[1].bits;
        used_cards |= hole_bits;
    }

    return sampleRemainingCardsBits(used_cards, num_cards, rng);
}

// Public wrapper for combining hole bits with board bits (used by multiway equity)
pub fn combineHoleBitsWithBoard(hole_bits: u64, board_bits: u64) poker.Hand {
    return combineCardsBits(hole_bits, board_bits);
}

// Helper function to generate bit combinations recursively
fn generateCombinationsBits(cards: []const poker.Card, cards_needed: u8, start_idx: usize, current_combination: u64, combinations: *[]u64, combination_idx: *usize) !void {
    if (cards_needed == 0) {
        combinations.*[combination_idx.*] = current_combination;
        combination_idx.* += 1;
        return;
    }

    for (start_idx..cards.len) |i| {
        if (cards.len - i < cards_needed) break;

        const new_combination = current_combination | cards[i].bits;
        try generateCombinationsBits(cards, cards_needed - 1, i + 1, new_combination, combinations, combination_idx);
    }
}

// Combine hole cards + board into 7-card Hand for evaluation
pub fn combineCards(hole_cards: [2]poker.Card, board_cards: []const poker.Card) poker.Hand {
    var hand = poker.Hand.init();
    for (hole_cards) |card| hand.addCard(card);
    for (board_cards) |card| hand.addCard(card);
    return hand;
}

// Extract individual cards not in used cards for enumeration
pub fn enumerateRemainingCards(used_cards: []const poker.Card, allocator: std.mem.Allocator) ![]poker.Card {
    const used_hand = poker.cardsToHand(used_cards);
    var cards = std.ArrayList(poker.Card).init(allocator);
    defer cards.deinit();

    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((used_hand.bits & card_bit) == 0) {
            const rank: u8 = @intCast((i / 4) + 2);
            const suit: u2 = @intCast(i % 4);
            try cards.append(poker.Card.init(rank, suit));
        }
    }

    return cards.toOwnedSlice();
}

// Get all possible combinations of n cards from remaining deck
pub fn enumerateCardCombinations(used_cards: []const poker.Card, num_cards: u8, allocator: std.mem.Allocator) ![][]poker.Card {
    const available_cards = try enumerateRemainingCards(used_cards, allocator);
    defer allocator.free(available_cards);

    if (num_cards > available_cards.len) {
        return error.NotEnoughCards;
    }

    const total_combinations = binomial(available_cards.len, num_cards);
    var combinations = try allocator.alloc([]poker.Card, total_combinations);
    var combination_idx: usize = 0;

    // Generate all combinations using recursion
    try generateCombinations(available_cards, num_cards, 0, &.{}, &combinations, &combination_idx, allocator);

    return combinations;
}

// Helper function to generate combinations recursively
fn generateCombinations(cards: []const poker.Card, cards_needed: u8, start_idx: usize, current_combination: []const poker.Card, combinations: *[][]poker.Card, combination_idx: *usize, allocator: std.mem.Allocator) !void {
    if (cards_needed == 0) {
        combinations.*[combination_idx.*] = try allocator.dupe(poker.Card, current_combination);
        combination_idx.* += 1;
        return;
    }

    for (start_idx..cards.len) |i| {
        if (cards.len - i < cards_needed) break;

        var new_combination = try allocator.alloc(poker.Card, current_combination.len + 1);
        defer allocator.free(new_combination);
        @memcpy(new_combination[0..current_combination.len], current_combination);
        new_combination[current_combination.len] = cards[i];

        try generateCombinations(cards, cards_needed - 1, i + 1, new_combination, combinations, combination_idx, allocator);
    }
}

// Calculate binomial coefficient (n choose k)
fn binomial(n: usize, k: usize) usize {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    var result: usize = 1;
    const min_k = @min(k, n - k);

    for (0..min_k) |i| {
        result = result * (n - i) / (i + 1);
    }

    return result;
}

// Tests
const testing = std.testing;

test "showdown evaluation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create two hands: pair vs high card (clear winner)
    const pair_cards = try poker.parseCards("AhAsKdQcJh2s3d", allocator);
    defer allocator.free(pair_cards);
    const high_card_cards = try poker.parseCards("KcQsJd8h7s4h5c", allocator);
    defer allocator.free(high_card_cards);

    const hands = [_]poker.Hand{ poker.cardsToHand(pair_cards), poker.cardsToHand(high_card_cards) };
    const result = try evaluateShowdown(&hands, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.winners.len == 1);
    try testing.expect(result.winners[0] == 0); // Pair wins
    try testing.expect(result.winning_rank == .pair);
}

test "card sampling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Use AA as hole cards
    const used_cards = poker.mustParseCards("AhAs");
    const sampled = try sampleRemainingCards(&used_cards, 5, rng, allocator);
    defer allocator.free(sampled);

    // Should have exactly 5 cards
    try testing.expect(sampled.len == 5);

    // Should not conflict with used cards (no AA in sampled)
    const used_hand = poker.cardsToHand(&used_cards);
    const sampled_hand = poker.cardsToHand(sampled);
    try testing.expect((used_hand.bits & sampled_hand.bits) == 0);
}

test "card combination" {
    // Use comptime parsing - no allocation needed
    const hole_cards = poker.mustParseCards("AhAs");
    const board_cards = poker.mustParseCards("KdQcJh2s3d");

    const combined = combineCards(hole_cards, &board_cards);
    try testing.expect(@popCount(combined.bits) == 7);
}

test "enumerate remaining cards" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const used_cards = poker.mustParseCards("AhAs");
    const remaining = try enumerateRemainingCards(&used_cards, allocator);
    defer allocator.free(remaining);

    try testing.expect(remaining.len == 50); // 52 - 2 used cards
}
