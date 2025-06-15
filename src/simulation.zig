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

// Sample remaining cards, avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: u64, num_cards: u8, rng: std.Random) u64 {
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

// Combine hole cards + board into 7-card Hand
pub fn combineCards(hole_cards: u64, board_cards: u64) poker.Hand {
    return poker.Hand{ .bits = hole_cards | board_cards };
}

// Extract individual cards from bitset for enumeration
pub fn enumerateRemainingCards(used_cards: u64, allocator: std.mem.Allocator) ![]poker.Card {
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

    return cards.toOwnedSlice();
}

// Get all possible combinations of n cards from remaining deck
pub fn enumerateCardCombinations(used_cards: u64, num_cards: u8, allocator: std.mem.Allocator) ![]u64 {
    const available_cards = try enumerateRemainingCards(used_cards, allocator);
    defer allocator.free(available_cards);

    if (num_cards > available_cards.len) {
        return error.NotEnoughCards;
    }

    const total_combinations = binomial(available_cards.len, num_cards);
    var combinations = try allocator.alloc(u64, total_combinations);
    var combination_idx: usize = 0;

    // Generate all combinations using recursion
    try generateCombinations(available_cards, num_cards, 0, 0, &combinations, &combination_idx);

    return combinations;
}

// Helper function to generate combinations recursively
fn generateCombinations(cards: []const poker.Card, cards_needed: u8, start_idx: usize, current_combination: u64, combinations: *[]u64, combination_idx: *usize) !void {
    if (cards_needed == 0) {
        combinations.*[combination_idx.*] = current_combination;
        combination_idx.* += 1;
        return;
    }

    for (start_idx..cards.len) |i| {
        if (cards.len - i < cards_needed) break;

        const new_combination = current_combination | cards[i].bits;
        try generateCombinations(cards, cards_needed - 1, i + 1, new_combination, combinations, combination_idx);
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

// Convert card array to hole cards bitset
pub fn cardsToHoleBits(cards: [2]poker.Card) u64 {
    return cards[0].bits | cards[1].bits;
}

// Convert board cards to bitset
pub fn boardToBits(board: []const poker.Card) u64 {
    var bits: u64 = 0;
    for (board) |card| {
        bits |= card.bits;
    }
    return bits;
}

// Tests
const testing = std.testing;

test "showdown evaluation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create two hands: pair vs high card (clear winner)
    const pair_hand = try poker.parseCards("AhAsKdQcJh2s3d");
    const high_card_hand = try poker.parseCards("KcQsJd8h7s4h5c");

    const hands = [_]poker.Hand{ pair_hand, high_card_hand };
    const result = try evaluateShowdown(&hands, allocator);
    defer result.deinit(allocator);

    try testing.expect(result.winners.len == 1);
    try testing.expect(result.winners[0] == 0); // Pair wins
    try testing.expect(result.winning_rank == .pair);
}

test "card sampling" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Use AA as hole cards
    const used = try poker.parseCards("AhAs");
    const sampled = sampleRemainingCards(used.bits, 5, rng);

    // Should have exactly 5 cards
    try testing.expect(@popCount(sampled) == 5);

    // Should not conflict with used cards
    try testing.expect((used.bits & sampled) == 0);
}

test "card combination" {
    const hole = try poker.parseCards("AhAs");
    const board = try poker.parseCards("KdQcJh2s3d");

    const combined = combineCards(hole.bits, board.bits);
    try testing.expect(@popCount(combined.bits) == 7);
}

test "enumerate remaining cards" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const used = try poker.parseCards("AhAs");
    const remaining = try enumerateRemainingCards(used.bits, allocator);
    defer allocator.free(remaining);

    try testing.expect(remaining.len == 50); // 52 - 2 used cards
}
