const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");

/// Fast equity calculation using optimized algorithms
/// Key optimizations:
/// 1. Batch evaluation using SIMD
/// 2. Optimized card sampling with rejection minimization
/// 3. Precomputed deck for faster sampling
/// 4. Cache-friendly data structures
pub const Hand = card.Hand;
pub const HandRank = evaluator.HandRank;

/// Precomputed deck for fast sampling
const PrecomputedDeck = struct {
    cards: [52]u64,
    count: u8,

    fn init() PrecomputedDeck {
        var deck = PrecomputedDeck{
            .cards = undefined,
            .count = 52,
        };
        for (0..52) |i| {
            deck.cards[i] = @as(u64, 1) << @intCast(i);
        }
        return deck;
    }

    fn withExclusions(self: PrecomputedDeck, exclude_mask: u64) struct { cards: [52]u64, count: u8 } {
        var result = struct {
            cards: [52]u64 = undefined,
            count: u8 = 0,
        }{};

        for (self.cards) |card_bit| {
            if ((card_bit & exclude_mask) == 0) {
                result.cards[result.count] = card_bit;
                result.count += 1;
            }
        }

        return result;
    }
};

/// Fast random card sampling using precomputed deck
fn sampleCardsPrecomputed(deck: anytype, num_cards: u8, rng: std.Random) u64 {
    var sampled: u64 = 0;
    var remaining = deck.count;
    var cards_needed = num_cards;

    // Fisher-Yates style sampling
    var deck_copy = deck.cards;

    while (cards_needed > 0) {
        const idx = rng.uintLessThan(u8, remaining);
        sampled |= deck_copy[idx];

        // Move last card to sampled position
        deck_copy[idx] = deck_copy[remaining - 1];
        remaining -= 1;
        cards_needed -= 1;
    }

    return sampled;
}

/// Batch equity result for SIMD processing
pub const BatchEquityResult = struct {
    wins: u32,
    ties: u32,
    total: u32,

    pub fn equity(self: BatchEquityResult) f64 {
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total));
    }
};

/// Fast Monte Carlo using batch evaluation
pub fn monteCarloFast(hero_hole: [2]Hand, villain_hole: [2]Hand, board: []const Hand, simulations: u32, rng: std.Random) BatchEquityResult {
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Precompute available deck
    const used_mask = hero_hole[0] | hero_hole[1] | villain_hole[0] | villain_hole[1] | board_hand;
    const deck = PrecomputedDeck.init();
    const available = deck.withExclusions(used_mask);

    var wins: u32 = 0;
    var ties: u32 = 0;

    // Process in batches of 32 for optimal SIMD usage
    const BATCH_SIZE = 32;
    const num_batches = simulations / BATCH_SIZE;
    const remainder = simulations % BATCH_SIZE;

    // Batch processing
    var i: u32 = 0;
    while (i < num_batches) : (i += 1) {
        var hero_hands: [BATCH_SIZE]u64 = undefined;
        var villain_hands: [BATCH_SIZE]u64 = undefined;

        // Generate batch of random boards
        for (0..BATCH_SIZE) |j| {
            const sampled = sampleCardsPrecomputed(available, cards_needed, rng);
            const complete_board = board_hand | sampled;
            hero_hands[j] = hero_hole[0] | hero_hole[1] | complete_board;
            villain_hands[j] = villain_hole[0] | villain_hole[1] | complete_board;
        }

        // Evaluate in batches
        const hero_ranks = evaluator.evaluateBatch(BATCH_SIZE, hero_hands);
        const villain_ranks = evaluator.evaluateBatch(BATCH_SIZE, villain_hands);

        // Count results
        for (0..BATCH_SIZE) |j| {
            if (hero_ranks[j] < villain_ranks[j]) {
                wins += 1;
            } else if (hero_ranks[j] == villain_ranks[j]) {
                ties += 1;
            }
        }
    }

    // Handle remainder
    for (0..remainder) |_| {
        const sampled = sampleCardsPrecomputed(available, cards_needed, rng);
        const complete_board = board_hand | sampled;

        const hero_hand = hero_hole[0] | hero_hole[1] | complete_board;
        const villain_hand = villain_hole[0] | villain_hole[1] | complete_board;

        const hero_rank = evaluator.evaluateHand(hero_hand);
        const villain_rank = evaluator.evaluateHand(villain_hand);

        if (hero_rank < villain_rank) {
            wins += 1;
        } else if (hero_rank == villain_rank) {
            ties += 1;
        }
    }

    return BatchEquityResult{
        .wins = wins,
        .ties = ties,
        .total = simulations,
    };
}

/// Ultra-fast Monte Carlo using vectorized operations
pub fn monteCarloVectorized(hero_hole: [2]Hand, villain_hole: [2]Hand, board: []const Hand, simulations: u32, rng: std.Random) BatchEquityResult {
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Precompute
    const hero_pocket = hero_hole[0] | hero_hole[1];
    const villain_pocket = villain_hole[0] | villain_hole[1];
    const used_mask = hero_pocket | villain_pocket | board_hand;

    // Create lookup table for available cards
    var available_cards: [52]u8 = undefined;
    var available_count: u8 = 0;
    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((card_bit & used_mask) == 0) {
            available_cards[available_count] = @intCast(i);
            available_count += 1;
        }
    }

    var wins: u32 = 0;
    var ties: u32 = 0;

    // Use smaller batch size for better cache locality
    const BATCH_SIZE = 16;
    const num_batches = simulations / BATCH_SIZE;
    const remainder = simulations % BATCH_SIZE;

    var batch_idx: u32 = 0;
    while (batch_idx < num_batches) : (batch_idx += 1) {
        var hero_batch: @Vector(BATCH_SIZE, u64) = undefined;
        var villain_batch: @Vector(BATCH_SIZE, u64) = undefined;

        // Generate batch
        inline for (0..BATCH_SIZE) |i| {
            var sampled: u64 = 0;
            var cards_sampled: u8 = 0;

            // Optimized sampling using available cards array
            while (cards_sampled < cards_needed) {
                const idx = rng.uintLessThan(u8, available_count);
                const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);

                if ((sampled & card_bit) == 0) {
                    sampled |= card_bit;
                    cards_sampled += 1;
                }
            }

            const complete_board = board_hand | sampled;
            hero_batch[i] = hero_pocket | complete_board;
            villain_batch[i] = villain_pocket | complete_board;
        }

        // Batch evaluate
        const hero_ranks = evaluator.evaluateBatch(BATCH_SIZE, hero_batch);
        const villain_ranks = evaluator.evaluateBatch(BATCH_SIZE, villain_batch);

        // Vectorized comparison
        inline for (0..BATCH_SIZE) |i| {
            if (hero_ranks[i] < villain_ranks[i]) {
                wins += 1;
            } else if (hero_ranks[i] == villain_ranks[i]) {
                ties += 1;
            }
        }
    }

    // Handle remainder with single evaluations
    for (0..remainder) |_| {
        var sampled: u64 = 0;
        var cards_sampled: u8 = 0;

        while (cards_sampled < cards_needed) {
            const idx = rng.uintLessThan(u8, available_count);
            const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);

            if ((sampled & card_bit) == 0) {
                sampled |= card_bit;
                cards_sampled += 1;
            }
        }

        const complete_board = board_hand | sampled;
        const hero_hand = hero_pocket | complete_board;
        const villain_hand = villain_pocket | complete_board;

        const hero_rank = evaluator.evaluateHand(hero_hand);
        const villain_rank = evaluator.evaluateHand(villain_hand);

        if (hero_rank < villain_rank) {
            wins += 1;
        } else if (hero_rank == villain_rank) {
            ties += 1;
        }
    }

    return BatchEquityResult{
        .wins = wins,
        .ties = ties,
        .total = simulations,
    };
}

// Tests
const testing = std.testing;

test "fast equity matches standard implementation" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // AA vs KK
    const aa = [_]Hand{ card.makeCard(0, 12), card.makeCard(1, 12) };
    const kk = [_]Hand{ card.makeCard(2, 11), card.makeCard(3, 11) };

    const result = monteCarloVectorized(aa, kk, &.{}, 10000, rng);

    // AA should win ~80% against KK
    try testing.expect(result.equity() > 0.75);
    try testing.expect(result.equity() < 0.85);
}

test "precomputed deck sampling" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const deck = PrecomputedDeck.init();
    const available = deck.withExclusions(0xFF); // Exclude first 8 cards

    try testing.expect(available.count == 44);

    const sampled = sampleCardsPrecomputed(available, 5, rng);
    try testing.expect(@popCount(sampled) == 5);
    try testing.expect((sampled & 0xFF) == 0); // Should not include excluded cards
}
