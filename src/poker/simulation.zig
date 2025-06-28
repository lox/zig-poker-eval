const std = @import("std");
const poker = @import("poker.zig");

pub const MultiplayerShowdownResult = struct {
    winners: []u8,
    winning_rank: poker.HandRank,

    pub fn deinit(self: MultiplayerShowdownResult, allocator: std.mem.Allocator) void {
        allocator.free(self.winners);
    }
};

const MAX_PLAYERS = 10; // Standard poker table limit

// Helper functions for card/bitmask conversion
fn cardSliceToBitmask(cards: []const poker.Card) u64 {
    var bitmask: u64 = 0;
    for (cards) |card| {
        bitmask |= card.bits;
    }
    return bitmask;
}

fn bitmaskToCardSlice(bitmask: u64, allocator: std.mem.Allocator) ![]poker.Card {
    var cards = std.ArrayList(poker.Card).init(allocator);
    defer cards.deinit();

    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((bitmask & card_bit) != 0) {
            const rank: u8 = @intCast((i / 4) + 2);
            const suit: u2 = @intCast(i % 4);
            try cards.append(poker.Card.init(rank, suit));
        }
    }

    return cards.toOwnedSlice();
}

// Core showdown evaluation - determines winners from multiple hands (optimized)
pub fn evaluateShowdown(hands: []const poker.Hand, allocator: std.mem.Allocator) !MultiplayerShowdownResult {
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

    return MultiplayerShowdownResult{
        .winners = winners,
        .winning_rank = best_rank,
    };
}

// Fast path for 2-player showdown (no allocation for common case) - use poker.ShowdownResult
pub fn evaluateShowdownHeadToHead(hand1: poker.Hand, hand2: poker.Hand) poker.ShowdownResult {
    return hand1.compareWith(hand2);
}

// Sample remaining cards, avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: []const poker.Card, num_cards: u8, rng: std.Random, allocator: std.mem.Allocator) ![]poker.Card {
    const used_bits = cardSliceToBitmask(used_cards);

    var sampled_cards: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        // Skip if card already used or sampled
        if ((used_bits & card_bit) != 0 or (sampled_cards & card_bit) != 0) {
            continue;
        }

        sampled_cards |= card_bit;
        cards_sampled += 1;
    }

    return bitmaskToCardSlice(sampled_cards, allocator);
}

// Combine hole cards + board into Hand for evaluation - delegates to poker module
pub fn combineCards(hole_cards: [2]poker.Card, board_cards: []const poker.Card) poker.Hand {
    return poker.Hand.fromHoleAndBoard(hole_cards, board_cards);
}

// Extract individual cards not in used cards for enumeration
pub fn enumerateRemainingCards(used_cards: []const poker.Card, allocator: std.mem.Allocator) ![]poker.Card {
    const used_bits = cardSliceToBitmask(used_cards);
    // All cards bitmask minus used cards
    const remaining_bits = ~used_bits & ((1 << 52) - 1);
    return bitmaskToCardSlice(remaining_bits, allocator);
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

    const hands = [_]poker.Hand{ combineCards(.{ pair_cards[0], pair_cards[1] }, pair_cards[2..]), combineCards(.{ high_card_cards[0], high_card_cards[1] }, high_card_cards[2..]) };
    const result = try evaluateShowdown(hands[0..], allocator);
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
    var used_bits: u64 = 0;
    for (used_cards) |card| {
        used_bits |= card.bits;
    }
    var sampled_bits: u64 = 0;
    for (sampled) |card| {
        sampled_bits |= card.bits;
    }
    try testing.expect((used_bits & sampled_bits) == 0);
}

test "card combination" {
    // Use comptime parsing - no allocation needed
    const hole_cards = poker.mustParseCards("AhAs");
    const board_cards = poker.mustParseCards("KdQcJh2s3d");

    const combined = combineCards(hole_cards, &board_cards);
    // Test that we can evaluate the combined hand
    const result = combined.evaluate();
    try testing.expect(@intFromEnum(result) >= 1 and @intFromEnum(result) <= 9);
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
