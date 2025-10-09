const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");

/// Equity calculation and Monte Carlo simulation for poker hands
/// This module provides head-to-head and multi-way equity calculations

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const HandRank = evaluator.HandRank;
pub const HandCategory = evaluator.HandCategory;

/// Basic equity result for head-to-head calculations
pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,

    pub fn winRate(self: EquityResult) f64 {
        return @as(f64, @floatFromInt(self.wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn tieRate(self: EquityResult) f64 {
        return @as(f64, @floatFromInt(self.ties)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn lossRate(self: EquityResult) f64 {
        const losses = self.total_simulations - self.wins - self.ties;
        return @as(f64, @floatFromInt(losses)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn equity(self: EquityResult) f64 {
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total_simulations));
    }
};

/// Hand category tracking for detailed analysis
pub const HandCategories = struct {
    high_card: u32 = 0,
    pair: u32 = 0,
    two_pair: u32 = 0,
    three_of_a_kind: u32 = 0,
    straight: u32 = 0,
    flush: u32 = 0,
    full_house: u32 = 0,
    four_of_a_kind: u32 = 0,
    straight_flush: u32 = 0,
    total: u32 = 0,

    pub fn addHand(self: *HandCategories, hand_rank: HandCategory) void {
        self.total += 1;
        switch (hand_rank) {
            .high_card => self.high_card += 1,
            .pair => self.pair += 1,
            .two_pair => self.two_pair += 1,
            .three_of_a_kind => self.three_of_a_kind += 1,
            .straight => self.straight += 1,
            .flush => self.flush += 1,
            .full_house => self.full_house += 1,
            .four_of_a_kind => self.four_of_a_kind += 1,
            .straight_flush => self.straight_flush += 1,
        }
    }

    pub fn percentage(self: HandCategories, count: u32) f64 {
        if (self.total == 0) return 0.0;
        return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(self.total)) * 100.0;
    }
};

/// Enhanced equity result with detailed statistics
pub const DetailedEquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,
    hand1_categories: HandCategories = HandCategories{},
    hand2_categories: HandCategories = HandCategories{},

    pub fn equity(self: DetailedEquityResult) f64 {
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn winRate(self: DetailedEquityResult) f64 {
        return @as(f64, @floatFromInt(self.wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn tieRate(self: DetailedEquityResult) f64 {
        return @as(f64, @floatFromInt(self.ties)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn lossRate(self: DetailedEquityResult) f64 {
        const losses = self.total_simulations - self.wins - self.ties;
        return @as(f64, @floatFromInt(losses)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    /// Calculate 95% confidence interval for equity
    pub fn confidenceInterval(self: DetailedEquityResult) struct { lower: f64, upper: f64 } {
        const equity_val = self.equity();
        const n = @as(f64, @floatFromInt(self.total_simulations));

        // Standard error for binomial proportion
        const se = @sqrt((equity_val * (1.0 - equity_val)) / n);

        // 95% confidence interval (±1.96 * SE)
        const margin = 1.96 * se;

        return .{
            .lower = @max(0.0, equity_val - margin),
            .upper = @min(1.0, equity_val + margin),
        };
    }
};

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

/// Head-to-head Monte Carlo equity calculation - optimized with SIMD batching
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
/// @param board Array of community cards (0-5 cards)
/// @param simulations Number of Monte Carlo simulations to run
/// @param rng Random number generator
/// @param allocator Memory allocator (unused but kept for API compatibility)
pub fn monteCarlo(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // No allocator needed for head-to-head equity
    _ = allocator; // Mark as unused

    // Precompute
    const used_mask = hero_hole_cards | villain_hole_cards | board_hand;

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
            hero_batch[i] = hero_hole_cards | complete_board;
            villain_batch[i] = villain_hole_cards | complete_board;
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
        const hero_hand = hero_hole_cards | complete_board;
        const villain_hand = villain_hole_cards | complete_board;

        const hero_rank = evaluator.evaluateHand(hero_hand);
        const villain_rank = evaluator.evaluateHand(villain_hand);

        if (hero_rank < villain_rank) {
            wins += 1;
        } else if (hero_rank == villain_rank) {
            ties += 1;
        }
    }

    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = simulations,
    };
}

/// Detailed Monte Carlo with hand category tracking
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
pub fn detailedMonteCarlo(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !DetailedEquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    var wins: u32 = 0;
    var ties: u32 = 0;
    var hand1_categories = HandCategories{};
    var hand2_categories = HandCategories{};

    // No allocator needed for head-to-head equity
    _ = allocator; // Mark as unused

    for (0..simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquityDirect(hero_hole_cards, villain_hole_cards, board_hand, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = hero_hole_cards | board_completion;
        const villain_hand = villain_hole_cards | board_completion;

        // Track hand categories
        const hero_rank = evaluator.evaluateHand(hero_hand);
        hand1_categories.addHand(evaluator.getHandCategory(hero_rank));
        const villain_rank = evaluator.evaluateHand(villain_hand);
        hand2_categories.addHand(evaluator.getHandCategory(villain_rank));

        const result = evaluateEquityShowdown(hero_hand, villain_hand);

        if (result != 0) {
            if (result > 0) {
                wins += 1;
            }
        } else {
            ties += 1;
        }
    }

    return DetailedEquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = simulations,
        .hand1_categories = hand1_categories,
        .hand2_categories = hand2_categories,
    };
}

/// Head-to-head exact equity calculation
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
pub fn exact(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Combine existing board cards
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    // Enumerate all possible board completions
    const board_completions = try enumerateEquityBoardCompletions(hero_hole_cards, villain_hole_cards, board, cards_needed, allocator);
    defer allocator.free(board_completions);

    var wins: u32 = 0;
    var ties: u32 = 0;

    for (board_completions) |board_completion| {
        // Create final hands with existing board + completion
        const complete_board = board_hand | board_completion;
        const hero_hand = hero_hole_cards | complete_board;
        const villain_hand = villain_hole_cards | complete_board;
        const result = evaluateEquityShowdown(hero_hand, villain_hand);

        if (result != 0) {
            if (result > 0) {
                wins += 1;
            }
        } else {
            ties += 1;
        }
    }

    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = @intCast(board_completions.len),
    };
}

/// Multi-way Monte Carlo equity calculation
/// @param hands Array of hole card bitmasks (each must contain exactly 2 cards)
pub fn multiway(hands: []const Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) ![]EquityResult {
    const num_players = hands.len;
    if (num_players < 2) return error.NotEnoughPlayers;

    // Validate all hole cards
    for (hands, 0..) |hole_cards, i| {
        if (card.countCards(hole_cards) != 2) return error.InvalidHoleCards;
        // Check for conflicts with previous hands
        for (hands[0..i]) |prev_cards| {
            if ((hole_cards & prev_cards) != 0) return error.ConflictingHoleCards;
        }
    }

    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Initialize results
    var results = try allocator.alloc(EquityResult, num_players);
    for (results) |*result| {
        result.* = EquityResult{ .wins = 0, .ties = 0, .total_simulations = simulations };
    }

    // Allocate buffers once outside the loop to avoid allocation overhead
    var final_hands = try allocator.alloc(Hand, num_players);
    defer allocator.free(final_hands);

    var winners: std.ArrayList(usize) = .empty;
    defer winners.deinit(allocator);

    for (0..simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquity(hands, board_hand, cards_needed, rng);

        // Reuse final_hands buffer
        for (hands, 0..) |hole_cards, i| {
            final_hands[i] = hole_cards | board_completion;
        }

        // Reuse winners ArrayList
        winners.clearRetainingCapacity();
        var best_rank: u16 = 65535; // Worst possible rank

        // SIMD batch evaluation for common player counts (2-8)
        // Falls back to scalar for edge cases (9+ players)
        if (num_players <= 8) {
            // Pad to batch size 8 for consistent SIMD evaluation
            var hands_batch: [8]u64 = [_]u64{0} ** 8;
            @memcpy(hands_batch[0..num_players], final_hands);

            const hands_vec: @Vector(8, u64) = hands_batch;
            const ranks_vec = evaluator.evaluateBatch(8, hands_vec);
            const ranks: [8]u16 = ranks_vec;

            // Find best rank and winners (only process actual players)
            for (ranks[0..num_players], 0..) |rank, i| {
                if (rank < best_rank) {
                    best_rank = rank;
                    winners.clearRetainingCapacity();
                    try winners.append(allocator, i);
                } else if (rank == best_rank) {
                    try winners.append(allocator, i);
                }
            }
        } else {
            // Scalar fallback for 9+ players
            for (final_hands, 0..) |hand, i| {
                const rank = evaluator.evaluateHand(hand);
                if (rank < best_rank) {
                    best_rank = rank;
                    winners.clearRetainingCapacity();
                    try winners.append(allocator, i);
                } else if (rank == best_rank) {
                    try winners.append(allocator, i);
                }
            }
        }

        if (winners.items.len == 1) {
            results[winners.items[0]].wins += 1;
        } else {
            // Split pot
            for (winners.items) |winner| {
                results[winner].ties += 1;
            }
        }
    }

    return results;
}

/// Hero vs field Monte Carlo equity (returns only hero's equity)
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_holes Array of villain hole card bitmasks
pub fn heroVsFieldMonteCarlo(hero_hole_cards: Hand, villain_holes: []const Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !f64 {
    // Build hands array with hero first
    var all_hands = try allocator.alloc(Hand, villain_holes.len + 1);
    defer allocator.free(all_hands);

    all_hands[0] = hero_hole_cards;
    @memcpy(all_hands[1..], villain_holes);

    const results = try multiway(all_hands, board, simulations, rng, allocator);
    defer allocator.free(results);

    return results[0].equity();
}

// === EQUITY-SPECIFIC HELPER FUNCTIONS ===

/// Fast path for head-to-head equity sampling (no array allocation)
fn sampleRemainingCardsHeadToHead(hero_hole_cards: Hand, villain_hole_cards: Hand, board: Hand, num_cards: u8, rng: std.Random) Hand {
    // Direct bit manipulation for maximum performance - use board directly
    const used_bits: u64 = hero_hole_cards | villain_hole_cards | board;

    var sampled_bits: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        if ((used_bits & card_bit) == 0 and (sampled_bits & card_bit) == 0) {
            sampled_bits |= card_bit;
            cards_sampled += 1;
        }
    }

    return sampled_bits;
}

/// Sample remaining cards for equity calculations (performance optimized)
fn sampleRemainingCardsForEquity(hands: []const Hand, board: Hand, num_cards: u8, rng: std.Random) Hand {
    // Fast path for head-to-head (most common case)
    if (hands.len == 2) {
        return sampleRemainingCardsHeadToHead(hands[0], hands[1], board, num_cards, rng);
    }

    // For multiway, use the same bit manipulation approach as head-to-head
    var used_bits: u64 = board;
    for (hands) |hole_cards| {
        used_bits |= hole_cards;
    }

    var sampled_bits: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        if ((used_bits & card_bit) == 0 and (sampled_bits & card_bit) == 0) {
            sampled_bits |= card_bit;
            cards_sampled += 1;
        }
    }

    return sampled_bits;
}

/// Direct sampling for head-to-head equity (used by detailedMonteCarlo)
fn sampleRemainingCardsForEquityDirect(hero_hole_cards: Hand, villain_hole_cards: Hand, board: Hand, num_cards: u8, rng: std.Random) Hand {
    return sampleRemainingCardsHeadToHead(hero_hole_cards, villain_hole_cards, board, num_cards, rng);
}

/// Evaluate equity showdown between two hands
pub fn evaluateEquityShowdown(hero_hand: Hand, villain_hand: Hand) i8 {
    const hands = @Vector(2, u64){ hero_hand, villain_hand };
    const ranks = evaluator.evaluateBatch(2, hands);
    const hero_rank = ranks[0];
    const villain_rank = ranks[1];
    // Lower rank numbers are better (0 = royal flush, 7461 = worst high card)
    return if (hero_rank < villain_rank) 1 else if (hero_rank > villain_rank) -1 else 0;
}

/// Enumerate all possible board completions for exact equity
fn enumerateEquityBoardCompletions(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, num_cards: u8, allocator: std.mem.Allocator) ![]Hand {
    // Special case: if board is complete, return single empty combination
    if (num_cards == 0) {
        const result = try allocator.alloc(Hand, 1);
        result[0] = 0; // Empty combination
        return result;
    }

    // Get all cards that are already in use
    var used_cards_bitmask: Hand = hero_hole_cards | villain_hole_cards;
    for (board) |board_card| {
        used_cards_bitmask |= board_card;
    }

    // Get remaining cards as individual card bits
    var remaining_cards: [52]u64 = undefined;
    var remaining_count: u8 = 0;
    for (0..52) |i| {
        const card_bit = @as(Hand, 1) << @intCast(i);
        if ((card_bit & used_cards_bitmask) == 0) {
            remaining_cards[remaining_count] = card_bit;
            remaining_count += 1;
        }
    }

    // Generate all combinations of num_cards from remaining cards
    var combinations: std.ArrayList(Hand) = .empty;
    errdefer combinations.deinit(allocator);

    // Use recursion to generate combinations
    var indices: [5]u8 = undefined;
    try generateCombinationsHelper(&combinations, allocator, &remaining_cards, remaining_count, num_cards, &indices, 0, 0);

    return try combinations.toOwnedSlice(allocator);
}

// Helper function to recursively generate combinations
fn generateCombinationsHelper(
    combinations: *std.ArrayList(Hand),
    allocator: std.mem.Allocator,
    cards: []const u64,
    total_cards: u8,
    num_needed: u8,
    indices: []u8,
    start_idx: u8,
    current_depth: u8,
) !void {
    if (current_depth == num_needed) {
        // We have a complete combination - OR all the selected cards together
        var combo: u64 = 0;
        for (0..num_needed) |i| {
            combo |= cards[indices[i]];
        }
        try combinations.append(allocator, combo);
        return;
    }

    // Generate combinations by selecting cards
    var i = start_idx;
    while (i <= total_cards - (num_needed - current_depth)) : (i += 1) {
        indices[current_depth] = i;
        try generateCombinationsHelper(combinations, allocator, cards, total_cards, num_needed, indices, i + 1, current_depth + 1);
    }
}

// === THREADED EQUITY CALCULATION ===

// Cache-line padded thread result to prevent false sharing
const ThreadResult = struct {
    wins: u32 align(64) = 0,
    ties: u32 = 0,
    total_simulations: u32 = 0,

    // Pad to cache line size (64 bytes on most architectures)
    _padding: [64 - 3 * @sizeOf(u32)]u8 = undefined,
};

const ThreadContext = struct {
    hero_hole_cards: Hand,
    villain_hole_cards: Hand,
    board: Hand,
    board_len: u8,
    simulations: u32,
    result: *ThreadResult,
    thread_id: u32,
    base_seed: u64,
    wait_group: *std.Thread.WaitGroup,
};

// Worker thread function
fn workerThread(ctx: *ThreadContext) void {
    defer ctx.wait_group.finish();

    // Initialize thread-local RNG with deterministic seed
    var prng = std.Random.DefaultPrng.init(ctx.base_seed + ctx.thread_id);
    const rng = prng.random();

    const cards_needed = 5 - ctx.board_len;

    var wins: u32 = 0;
    var ties: u32 = 0;

    // Run simulations assigned to this thread
    for (0..ctx.simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquityDirect(ctx.hero_hole_cards, ctx.villain_hole_cards, ctx.board, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = ctx.hero_hole_cards | board_completion;
        const villain_hand = ctx.villain_hole_cards | board_completion;
        const result = evaluateEquityShowdown(hero_hand, villain_hand);

        if (result != 0) {
            if (result > 0) {
                wins += 1;
            }
        } else {
            ties += 1;
        }
    }

    // Store results (each thread writes to its own cache line)
    ctx.result.wins = wins;
    ctx.result.ties = ties;
    ctx.result.total_simulations = ctx.simulations;
}

/// Multi-threaded Monte Carlo equity calculation
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
pub fn threaded(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, base_seed: u64, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;
    // Get optimal thread count (but cap at reasonable limit)
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const sims_per_thread = simulations / thread_count;
    const remaining_sims = simulations % thread_count;

    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const board_len = @as(u8, @intCast(board.len));

    // Allocate cache-line padded results
    var thread_results = try allocator.alloc(ThreadResult, thread_count);
    defer allocator.free(thread_results);

    // Initialize results
    for (thread_results) |*result| {
        result.* = ThreadResult{};
    }

    // Allocate thread contexts
    var contexts = try allocator.alloc(ThreadContext, thread_count);
    defer allocator.free(contexts);

    var wait_group = std.Thread.WaitGroup{};

    // Spawn worker threads
    for (0..thread_count) |i| {
        const thread_sims = sims_per_thread + (if (i == 0) remaining_sims else 0);

        contexts[i] = ThreadContext{
            .hero_hole_cards = hero_hole_cards,
            .villain_hole_cards = villain_hole_cards,
            .board = board_hand,
            .board_len = board_len,
            .simulations = thread_sims,
            .result = &thread_results[i],
            .thread_id = @intCast(i),
            .base_seed = base_seed,
            .wait_group = &wait_group,
        };

        wait_group.start();
        _ = try std.Thread.spawn(.{}, workerThread, .{&contexts[i]});
    }

    // Wait for all threads to complete
    wait_group.wait();

    // Aggregate results (fast, single-threaded)
    var total_wins: u32 = 0;
    var total_ties: u32 = 0;
    var total_sims: u32 = 0;

    for (thread_results) |result| {
        total_wins += result.wins;
        total_ties += result.ties;
        total_sims += result.total_simulations;
    }

    return EquityResult{
        .wins = total_wins,
        .ties = total_ties,
        .total_simulations = total_sims,
    };
}

// Tests
const testing = std.testing;

test "AA vs KK equity should be ~80%" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace); // AcAd
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king); // KhKs

    const result = try monteCarlo(aa, kk, &.{}, 100000, rng, allocator);

    // AA should beat KK roughly 80% of the time
    try testing.expect(result.equity() > 0.75);
    try testing.expect(result.equity() < 0.85);
}

test "AA vs 22 equity should be ~85%" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace); // AcAd
    const twos = card.makeCard(.hearts, .two) | card.makeCard(.spades, .two); // 2h2s

    const result = try monteCarlo(aa, twos, &.{}, 100000, rng, allocator);

    // AA should dominate 22
    try testing.expect(result.equity() > 0.80);
    try testing.expect(result.equity() < 0.90);
}

test "hand evaluation sanity check" {
    // Test with a very simple case that should clearly work
    const aa_hole = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace); // AcAd
    const two_hole = card.makeCard(.hearts, .two) | card.makeCard(.spades, .two); // 2h2s

    // Simple board that doesn't improve either
    const board = [_]Hand{
        card.makeCard(.hearts, .seven), // 7h
        card.makeCard(.spades, .eight), // 8s
        card.makeCard(.clubs, .nine), // 9c
        card.makeCard(.diamonds, .three), // 3d
        card.makeCard(.hearts, .four), // 4h
    };

    const aa_final = aa_hole | (board[0] | board[1] | board[2] | board[3] | board[4]);
    const two_final = two_hole | (board[0] | board[1] | board[2] | board[3] | board[4]);

    const result = evaluateEquityShowdown(aa_final, two_final);

    // AA should definitely beat 22
    try testing.expect(result != 0); // Not a tie
    try testing.expect(result == 1); // AA wins
}

test "exact equity AA vs KK on turn" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Board through turn (only 44 possible rivers)
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
        card.makeCard(.clubs, .two),
    };

    const result = try exact(aa, kk, &board, allocator);

    // AA should beat KK with this board
    try testing.expect(result.total_simulations == 44);
    try testing.expect(result.equity() > 0.90);
}

test "exact equity with complete board" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Complete board - AA wins with set
    const board = [_]Hand{
        card.makeCard(.spades, .ace),
        card.makeCard(.diamonds, .seven),
        card.makeCard(.hearts, .two),
        card.makeCard(.clubs, .three),
        card.makeCard(.spades, .five),
    };

    const result = try exact(aa, kk, &board, allocator);

    // With complete board, only 1 scenario, AA wins 100%
    try testing.expect(result.total_simulations == 1);
    try testing.expect(result.wins == 1);
    try testing.expect(result.equity() == 1.0);
}

test "cache line padding" {
    // Verify ThreadResult is properly padded
    try testing.expect(@sizeOf(ThreadResult) == 64);
    try testing.expect(@alignOf(ThreadResult) == 64);
}

test "exact equity with conflicting cards should error" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const aa_conflict = card.makeCard(.clubs, .ace) | card.makeCard(.hearts, .ace);

    try testing.expectError(error.ConflictingHoleCards, exact(aa, aa_conflict, &.{}, allocator));
}

test "exact equity validates hole card count" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const one_card = card.makeCard(.clubs, .ace);
    const two_cards = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    try testing.expectError(error.InvalidHeroHoleCards, exact(one_card, two_cards, &.{}, allocator));
}

test "monteCarlo equity validates hole card count" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const one_card = card.makeCard(.clubs, .ace);
    const two_cards = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    try testing.expectError(error.InvalidHeroHoleCards, monteCarlo(one_card, two_cards, &.{}, 100, rng, allocator));
}

test "EquityResult helper methods" {
    const result = EquityResult{
        .wins = 80,
        .ties = 10,
        .total_simulations = 100,
    };

    try testing.expect(result.winRate() == 0.80);
    try testing.expect(result.tieRate() == 0.10);
    try testing.expect(result.lossRate() == 0.10);
    try testing.expect(result.equity() == 0.85); // 80 + 10*0.5 = 85%
}

test "DetailedEquityResult confidence interval" {
    const result = DetailedEquityResult{
        .wins = 500,
        .ties = 0,
        .total_simulations = 1000,
    };

    const ci = result.confidenceInterval();

    // 50% equity, should have reasonable confidence bounds
    try testing.expect(ci.lower >= 0.0);
    try testing.expect(ci.upper <= 1.0);
    try testing.expect(ci.lower < 0.5);
    try testing.expect(ci.upper > 0.5);
}

test "exact equity with flop" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Flop - this will enumerate C(45, 2) = 990 turn/river combinations
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
    };

    const result = try exact(aa, kk, &board, allocator);

    try testing.expect(result.total_simulations == 990);
    // AA should beat KK most of the time (but not as strongly as with a turn)
    try testing.expect(result.equity() > 0.80);
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
