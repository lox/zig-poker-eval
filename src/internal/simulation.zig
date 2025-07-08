const std = @import("std");

pub const MultiplayerShowdownResult = struct {
    winners: []u8,
    winning_rank: evaluator.HandRank,

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
            const suit: card.Suit = @enumFromInt(i / 13); // Suit from bit position
            const rank: card.Rank = @enumFromInt(i % 13); // Rank from bit position
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
    var ranks: [MAX_PLAYERS]evaluator.HandRank = undefined;
    var best_rank: evaluator.HandRank = 65535; // Worst possible rank

    // Evaluate all hands and find best rank in single pass
    for (hands, 0..) |h, i| {
        ranks[i] = evaluator.evaluateHand(h);
        if (ranks[i] < best_rank) {
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
// Returns Hand (CardSet) with num_cards randomly sampled
pub fn sampleRemainingCards(used_cards: card.Hand, num_cards: u8, rng: std.Random) card.Hand {
    var sampled_cards: card.Hand = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(card.Hand, 1) << @intCast(card_idx);

        // Skip if card already used or sampled
        if ((used_cards & card_bit) != 0 or (sampled_cards & card_bit) != 0) {
            continue;
        }

        sampled_cards |= card_bit;
        cards_sampled += 1;
    }

    return sampled_cards;
}

// Extract individual cards not in used cards for enumeration
// Get remaining cards as CardSet (bitmask)
pub fn enumerateRemainingCards(used_cards: card.Hand) card.Hand {
    // All cards bitmask minus used cards
    const all_cards_mask = (@as(card.Hand, 1) << 52) - 1;
    return ~used_cards & all_cards_mask;
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
    const ace_hearts = card.makeCard(.hearts, .ace); // Ah
    const ace_spades = card.makeCard(.spades, .ace); // As
    const king_diamonds = card.makeCard(.diamonds, .king); // Kd
    const queen_clubs = card.makeCard(.clubs, .queen); // Qc
    const jack_hearts = card.makeCard(.hearts, .jack); // Jh
    const two_spades = card.makeCard(.spades, .two); // 2s
    const three_diamonds = card.makeCard(.diamonds, .three); // 3d

    // Create the 7-card hand: Ah As Kd Qc Jh 2s 3d
    const hand_with_pair = ace_hearts | ace_spades | king_diamonds | queen_clubs | jack_hearts | two_spades | three_diamonds;

    // Test with the evaluator
    const raw_rank = evaluator.evaluateHand(hand_with_pair);
    const category = evaluator.getHandCategory(raw_rank);

    // This should be ONE PAIR, not two pair - should pass now with corrected boundaries
    try testing.expect(category == .pair);

    const hand1 = hand_with_pair;
    const hand2 = king_diamonds | queen_clubs | jack_hearts | two_spades | three_diamonds | card.makeCard(.clubs, .eight) | card.makeCard(.spades, .seven); // High card hand

    const hands = [_]card.Hand{ hand1, hand2 };
    const result = try evaluateShowdown(hands[0..], allocator);
    defer result.deinit(allocator);

    try testing.expect(result.winners.len == 1);
    try testing.expect(result.winners[0] == 0); // Pair wins over high card
    const winning_category = evaluator.getHandCategory(result.winning_rank);
    try testing.expect(winning_category == .pair);
}

test "card sampling" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    _ = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Use AA as hole cards (CardSet approach)
    const used_cards = hand.parseHand("AhAs");
    const sampled = sampleRemainingCards(used_cards, 5, rng);

    // Should have exactly 5 cards
    try testing.expect(card.countCards(sampled) == 5);

    // Should not conflict with used cards (no overlap)
    try testing.expect((used_cards & sampled) == 0);
}

test "card combination" {
    // Use comptime parsing with CardSet approach
    const hole_cards = hand.parseHand("AhAs");
    const board_cards = hand.parseHand("KdQcJh2s3d");

    const combined = hole_cards | board_cards;

    // Verify we have 7 cards total
    try testing.expect(card.countCards(combined) == 7);

    // Test that we can evaluate the combined hand
    const result = evaluator.evaluateHand(combined);
    try testing.expect(result >= 1 and result <= 7462); // Hand rank values
}

test "enumerate remaining cards" {
    const used_cards = hand.parseHand("AhAs");
    const remaining = enumerateRemainingCards(used_cards);

    try testing.expect(card.countCards(remaining) == 50); // 52 - 2 used cards

    // Verify no overlap with used cards
    try testing.expect((used_cards & remaining) == 0);
}
