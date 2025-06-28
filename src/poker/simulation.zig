const std = @import("std");
const card = @import("../card/mod.zig");
const poker = @import("poker.zig");
const evaluator = @import("../evaluator/mod.zig");

pub const MultiplayerShowdownResult = struct {
    winners: []u8,
    winning_rank: poker.HandRank,

    pub fn deinit(self: MultiplayerShowdownResult, allocator: std.mem.Allocator) void {
        allocator.free(self.winners);
    }
};

const MAX_PLAYERS = 10; // Standard poker table limit

// Helper functions for card/bitmask conversion
fn cardSliceToBitmask(cards: []const card.Hand) u64 {
    var bitmask: u64 = 0;
    for (cards) |hand_card| {
        bitmask |= hand_card;
    }
    return bitmask;
}

fn bitmaskToCardSlice(bitmask: u64, allocator: std.mem.Allocator) ![]card.Hand {
    var cards = std.ArrayList(card.Hand).init(allocator);
    defer cards.deinit();

    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((bitmask & card_bit) != 0) {
            const rank: u8 = @intCast(i / 4); // card rank format (0-12)
            const suit: u2 = @intCast(i % 4);
            try cards.append(card.makeCard(suit, rank));
        }
    }

    return cards.toOwnedSlice();
}

// Core showdown evaluation - determines winners from multiple hands (optimized)
pub fn evaluateShowdown(hands: []const card.Hand, allocator: std.mem.Allocator) !MultiplayerShowdownResult {
    if (hands.len == 0) return error.NoHands;
    if (hands.len > MAX_PLAYERS) return error.TooManyPlayers;

    // Use stack allocation for ranks (no heap allocation!)
    var ranks: [MAX_PLAYERS]poker.HandRank = undefined;
    var best_rank: poker.HandRank = .high_card;

    // Evaluate all hands and find best rank in single pass
    for (hands, 0..) |hand, i| {
        ranks[i] = poker.convertEvaluatorRank(evaluator.evaluateHand(hand));
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
pub fn evaluateShowdownHeadToHead(hand1: card.Hand, hand2: card.Hand) i8 {
    const rank1 = evaluator.evaluateHand(hand1);
    const rank2 = evaluator.evaluateHand(hand2);
    return if (rank1 > rank2) 1 else if (rank1 < rank2) -1 else 0;
}

// Sample remaining cards, avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: []const card.Hand, num_cards: u8, rng: std.Random, allocator: std.mem.Allocator) ![]card.Hand {
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
pub fn combineCards(hole_cards: [2]card.Hand, board_cards: []const card.Hand) card.Hand {
    return card.fromHoleAndBoard(hole_cards, board_cards);
}

// Extract individual cards not in used cards for enumeration
pub fn enumerateRemainingCards(used_cards: []const card.Hand, allocator: std.mem.Allocator) ![]card.Hand {
    const used_bits = cardSliceToBitmask(used_cards);
    // All cards bitmask minus used cards
    const remaining_bits = ~used_bits & ((1 << 52) - 1);
    return bitmaskToCardSlice(remaining_bits, allocator);
}

// Get all possible combinations of n cards from remaining deck
pub fn enumerateCardCombinations(used_cards: []const card.Hand, num_cards: u8, allocator: std.mem.Allocator) ![][]card.Hand {
    const available_cards = try enumerateRemainingCards(used_cards, allocator);
    defer allocator.free(available_cards);

    if (num_cards > available_cards.len) {
        return error.NotEnoughCards;
    }

    const total_combinations = binomial(available_cards.len, num_cards);
    var combinations = try allocator.alloc([]card.Hand, total_combinations);
    var combination_idx: usize = 0;

    // Generate all combinations using recursion
    try generateCombinations(available_cards, num_cards, 0, &.{}, &combinations, &combination_idx, allocator);

    return combinations;
}

// Helper function to generate combinations recursively
fn generateCombinations(cards: []const card.Hand, cards_needed: u8, start_idx: usize, current_combination: []const card.Hand, combinations: *[][]card.Hand, combination_idx: *usize, allocator: std.mem.Allocator) !void {
    if (cards_needed == 0) {
        combinations.*[combination_idx.*] = try allocator.dupe(card.Hand, current_combination);
        combination_idx.* += 1;
        return;
    }

    for (start_idx..cards.len) |i| {
        if (cards.len - i < cards_needed) break;

        var new_combination = try allocator.alloc(card.Hand, current_combination.len + 1);
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

    // Let's debug this step by step
    // First, let's create a simple pair of aces hand and check it
    const ace_hearts = card.makeCard(2, 12); // Ah (hearts=2, ace=12)
    const ace_spades = card.makeCard(3, 12); // As (spades=3, ace=12)
    const king_diamonds = card.makeCard(1, 11); // Kd
    const queen_clubs = card.makeCard(0, 10); // Qc
    const jack_hearts = card.makeCard(2, 9); // Jh
    const two_spades = card.makeCard(3, 0); // 2s
    const three_diamonds = card.makeCard(1, 1); // 3d
    
    // Create the 7-card hand: Ah As Kd Qc Jh 2s 3d
    const hand_with_pair = ace_hearts | ace_spades | king_diamonds | queen_clubs | jack_hearts | two_spades | three_diamonds;
    
    // Test with the evaluator 
    const raw_rank = evaluator.evaluateHand(hand_with_pair);
    const converted_rank = poker.convertEvaluatorRank(raw_rank);
    
    // This should be ONE PAIR, not two pair - should pass now with corrected boundaries
    try testing.expect(converted_rank == .pair);
    
    const hand1 = hand_with_pair;
    const hand2 = king_diamonds | queen_clubs | jack_hearts | two_spades | three_diamonds | card.makeCard(0, 6) | card.makeCard(3, 5); // High card hand
    
    
    const hands = [_]card.Hand{ hand1, hand2 };
    const result = try evaluateShowdown(hands[0..], allocator);
    defer result.deinit(allocator);

    try testing.expect(result.winners.len == 1);
    try testing.expect(result.winners[0] == 0); // Pair wins over high card
    try testing.expect(result.winning_rank == .pair);
}

test "card sampling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Use AA as hole cards
    const used_cards = card.mustParseCards("AhAs");
    const sampled = try sampleRemainingCards(&used_cards, 5, rng, allocator);
    defer allocator.free(sampled);

    // Should have exactly 5 cards
    try testing.expect(sampled.len == 5);

    // Should not conflict with used cards (no AA in sampled)
    var used_bits: u64 = 0;
    for (used_cards) |used_card| {
        used_bits |= used_card;
    }
    var sampled_bits: u64 = 0;
    for (sampled) |sampled_card| {
        sampled_bits |= sampled_card;
    }
    try testing.expect((used_bits & sampled_bits) == 0);
}

test "card combination" {
    // Use comptime parsing - no allocation needed
    const hole_cards = card.mustParseCards("AhAs");
    const board_cards = card.mustParseCards("KdQcJh2s3d");

    const combined = combineCards(hole_cards, &board_cards);
    // Test that we can evaluate the combined hand
    const result = card.evaluate(combined);
    try testing.expect(result >= 1 and result <= 7462); // Hand rank values
}

test "enumerate remaining cards" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const used_cards = card.mustParseCards("AhAs");
    const remaining = try enumerateRemainingCards(&used_cards, allocator);
    defer allocator.free(remaining);

    try testing.expect(remaining.len == 50); // 52 - 2 used cards
}
