const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");
const deck = @import("deck");

/// Equity calculation and Monte Carlo simulation for poker hands
/// This module provides head-to-head and multi-way equity calculations

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const HandRank = evaluator.HandRank;
pub const HandCategory = evaluator.HandCategory;

/// Unified equity result for all calculation types
/// Supports both basic and detailed (category-tracking) calculations
pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,
    hand1_categories: ?HandCategories = null,
    hand2_categories: ?HandCategories = null,
    method: Method = .monte_carlo,

    pub const Method = enum {
        monte_carlo,
        exact,
    };

    pub fn winRate(self: EquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn tieRate(self: EquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ties)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn lossRate(self: EquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        const losses = self.total_simulations - self.wins - self.ties;
        return @as(f64, @floatFromInt(losses)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn equity(self: EquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total_simulations));
    }

    /// Calculate 95% confidence interval for Monte Carlo equity
    /// Returns null for exact calculations or empty results
    pub fn confidenceInterval(self: EquityResult) ?struct { lower: f64, upper: f64 } {
        // Only valid for Monte Carlo results with simulations
        if (self.total_simulations == 0 or self.method == .exact) return null;

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

/// Validate board cards for conflicts and duplicates
fn validateBoard(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand) !Hand {
    if (board.len > 5) return error.InvalidBoardLength;

    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    // Check for duplicates in board
    if (card.countCards(board_hand) != board.len) return error.DuplicateBoardCards;

    // Check for conflicts with hole cards
    if ((board_hand & hero_hole_cards) != 0) return error.BoardConflictsWithHeroHoleCards;
    if ((board_hand & villain_hole_cards) != 0) return error.BoardConflictsWithVillainHoleCards;

    return board_hand;
}

/// Internal Monte Carlo implementation with comptime category tracking and SIMD batching
fn monteCarloImpl(comptime track_categories: bool, hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;

    // Validate board
    const board_hand = try validateBoard(hero_hole_cards, villain_hole_cards, board);
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // No allocator needed for head-to-head equity
    _ = allocator;

    // Precompute available cards
    const used_mask = hero_hole_cards | villain_hole_cards | board_hand;
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

    // Storage for categories (always allocated, but only used if track_categories is true)
    var hand1_cat_storage = HandCategories{};
    var hand2_cat_storage = HandCategories{};

    // Use SIMD batching for performance
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

        // Vectorized comparison with optional category tracking (comptime branching)
        inline for (0..BATCH_SIZE) |i| {
            if (track_categories) {
                hand1_cat_storage.addHand(evaluator.getHandCategory(hero_ranks[i]));
                hand2_cat_storage.addHand(evaluator.getHandCategory(villain_ranks[i]));
            }

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

        if (track_categories) {
            hand1_cat_storage.addHand(evaluator.getHandCategory(hero_rank));
            hand2_cat_storage.addHand(evaluator.getHandCategory(villain_rank));
        }

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
        .hand1_categories = if (track_categories) hand1_cat_storage else null,
        .hand2_categories = if (track_categories) hand2_cat_storage else null,
    };
}

/// Head-to-head Monte Carlo equity calculation - optimized with SIMD batching
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
/// @param board Array of community cards (0-5 cards)
/// @param simulations Number of Monte Carlo simulations to run
/// @param rng Random number generator
/// @param allocator Memory allocator (unused but kept for API compatibility)
pub fn monteCarlo(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    return monteCarloImpl(false, hero_hole_cards, villain_hole_cards, board, simulations, rng, allocator);
}

/// Monte Carlo with hand category tracking
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
pub fn monteCarloWithCategories(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    return monteCarloImpl(true, hero_hole_cards, villain_hole_cards, board, simulations, rng, allocator);
}

/// Internal exact equity implementation with comptime category tracking
fn exactImpl(comptime track_categories: bool, hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;

    // Validate board
    const board_hand = try validateBoard(hero_hole_cards, villain_hole_cards, board);
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Experiment 26: Use greedy batch cascade for turn→river case (1 card needed)
    if (cards_needed == 1) {
        return exactTurnStreaming(track_categories, hero_hole_cards, villain_hole_cards, board_hand);
    }

    // Enumerate all possible board completions
    const board_completions = try enumerateEquityBoardCompletions(hero_hole_cards, villain_hole_cards, board, cards_needed, allocator);
    defer allocator.free(board_completions);

    var wins: u32 = 0;
    var ties: u32 = 0;

    // Storage for categories (always allocated, but only used if track_categories is true)
    var hand1_cat_storage = HandCategories{};
    var hand2_cat_storage = HandCategories{};

    // SIMD batching for board evaluation (Experiment 17)
    const BATCH_SIZE = 32;
    const num_batches = board_completions.len / BATCH_SIZE;

    var batch_idx: usize = 0;
    while (batch_idx < num_batches) : (batch_idx += 1) {
        var hero_batch: @Vector(BATCH_SIZE, u64) = undefined;
        var villain_batch: @Vector(BATCH_SIZE, u64) = undefined;

        // Prepare batch of full 7-card hands
        inline for (0..BATCH_SIZE) |j| {
            const complete_board = board_hand | board_completions[batch_idx * BATCH_SIZE + j];
            hero_batch[j] = hero_hole_cards | complete_board;
            villain_batch[j] = villain_hole_cards | complete_board;
        }

        // SIMD batch evaluation
        const hero_ranks = evaluator.evaluateBatch(BATCH_SIZE, hero_batch);
        const villain_ranks = evaluator.evaluateBatch(BATCH_SIZE, villain_batch);

        // Compare results and track categories (comptime branching)
        inline for (0..BATCH_SIZE) |j| {
            if (track_categories) {
                hand1_cat_storage.addHand(evaluator.getHandCategory(hero_ranks[j]));
                hand2_cat_storage.addHand(evaluator.getHandCategory(villain_ranks[j]));
            }

            if (hero_ranks[j] < villain_ranks[j]) {
                wins += 1;
            } else if (hero_ranks[j] == villain_ranks[j]) {
                ties += 1;
            }
        }
    }

    // Handle remainder with board context (Experiment 16 approach)
    const remainder_start = num_batches * BATCH_SIZE;
    for (board_completions[remainder_start..]) |board_completion| {
        const complete_board = board_hand | board_completion;
        const ctx = evaluator.initBoardContext(complete_board);

        // Track hand categories (comptime branching)
        if (track_categories) {
            const hero_rank = evaluator.evaluateHoleWithContext(&ctx, hero_hole_cards);
            hand1_cat_storage.addHand(evaluator.getHandCategory(hero_rank));
            const villain_rank = evaluator.evaluateHoleWithContext(&ctx, villain_hole_cards);
            hand2_cat_storage.addHand(evaluator.getHandCategory(villain_rank));
        }

        const result = evaluator.evaluateShowdownWithContext(&ctx, hero_hole_cards, villain_hole_cards);

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
        .hand1_categories = if (track_categories) hand1_cat_storage else null,
        .hand2_categories = if (track_categories) hand2_cat_storage else null,
        .method = .exact,
    };
}

/// Head-to-head exact equity calculation
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
pub fn exact(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    return exactImpl(false, hero_hole_cards, villain_hole_cards, board, allocator);
}

/// Exact equity calculation with hand category tracking
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param villain_hole_cards Combined bitmask of villain's exactly 2 hole cards
/// @param board Array of community cards (0-5 cards)
/// @param allocator Memory allocator for board enumeration
pub fn exactWithCategories(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    return exactImpl(true, hero_hole_cards, villain_hole_cards, board, allocator);
}

/// Exact equity calculation against random opponent (all possible villain hands and boards)
/// Optimized with board-first enumeration and SIMD batching for performance
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param allocator Memory allocator for temporary buffers
pub fn exactVsRandom(hero_hole_cards: Hand, allocator: std.mem.Allocator) !EquityResult {
    // Validate hole cards
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;

    var wins: u64 = 0;
    var losses: u64 = 0;
    var ties: u64 = 0;

    // Build available cards (all except hero's 2 cards)
    var available: [50]u8 = undefined;
    var available_count: u8 = 0;
    for (0..52) |card_idx| {
        const card_bit = @as(u64, 1) << @intCast(card_idx);
        if ((hero_hole_cards & card_bit) == 0) {
            available[available_count] = @intCast(card_idx);
            available_count += 1;
        }
    }

    // Pre-allocate villain hands array for batch processing
    const max_villain_hands = 1225; // C(50,2)
    var all_villain_holes: [max_villain_hands]Hand = undefined;
    var villain_holes_buffer: [max_villain_hands]Hand = undefined;
    var hero_holes_buffer: [max_villain_hands]Hand = undefined;
    var results_buffer: [max_villain_hands]i8 = undefined;

    // Generate all villain hands once and store in all_villain_holes
    var villain_count: usize = 0;
    for (0..available_count - 1) |v1| {
        for (v1 + 1..available_count) |v2| {
            const villain_card1 = @as(u64, 1) << @intCast(available[v1]);
            const villain_card2 = @as(u64, 1) << @intCast(available[v2]);
            all_villain_holes[villain_count] = villain_card1 | villain_card2;
            villain_count += 1;
        }
    }

    // Enumerate all boards: C(48,5)
    // For each board, batch-evaluate against all villain hands
    for (0..available_count - 4) |b1| {
        const board1 = @as(u64, 1) << @intCast(available[b1]);

        for (b1 + 1..available_count - 3) |b2| {
            const board2 = @as(u64, 1) << @intCast(available[b2]);

            for (b2 + 1..available_count - 2) |b3| {
                const board3 = @as(u64, 1) << @intCast(available[b3]);

                for (b3 + 1..available_count - 1) |b4| {
                    const board4 = @as(u64, 1) << @intCast(available[b4]);

                    for (b4 + 1..available_count) |b5| {
                        const board5 = @as(u64, 1) << @intCast(available[b5]);

                        const board = board1 | board2 | board3 | board4 | board5;

                        // Create board context once for this board (board only, not hole cards)
                        const ctx = evaluator.initBoardContext(board);

                        // Filter villain hands that don't conflict with board
                        // Copy valid hands from all_villain_holes to villain_holes_buffer
                        var valid_villain_count: usize = 0;
                        for (0..villain_count) |v| {
                            const villain_hole = all_villain_holes[v];
                            if ((villain_hole & board) == 0) {
                                villain_holes_buffer[valid_villain_count] = villain_hole;
                                hero_holes_buffer[valid_villain_count] = hero_hole_cards;
                                valid_villain_count += 1;
                            }
                        }

                        // Batch evaluate all valid villain hands
                        if (valid_villain_count > 0) {
                            evaluator.evaluateShowdownBatch(
                                &ctx,
                                hero_holes_buffer[0..valid_villain_count],
                                villain_holes_buffer[0..valid_villain_count],
                                results_buffer[0..valid_villain_count],
                            );

                            // Accumulate results
                            for (results_buffer[0..valid_villain_count]) |result| {
                                if (result > 0) {
                                    wins += 1;
                                } else if (result < 0) {
                                    losses += 1;
                                } else {
                                    ties += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Mark allocator as used (even though we don't allocate anything)
    _ = allocator;

    const total = wins + losses + ties;
    return EquityResult{
        .wins = @intCast(wins),
        .ties = @intCast(ties),
        .total_simulations = @intCast(total),
    };
}

/// Monte Carlo equity calculation against a uniformly random opponent hand.
/// Samples random opponent hole cards from remaining deck, runs out the board.
///
/// Example: AA on flop Kd7c2s - what's our equity vs a random hand?
/// @param hero_hole_cards Combined bitmask of hero's exactly 2 hole cards
/// @param board Array of community cards (0-5 cards)
/// @param simulations Number of Monte Carlo iterations
/// @param rng Random number generator
/// @param allocator Memory allocator (unused but kept for API compatibility)
pub fn equityVsRandom(hero_hole_cards: Hand, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    _ = allocator;

    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (board.len > 5) return error.InvalidBoardLength;

    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    if (card.countCards(board_hand) != board.len) return error.DuplicateBoardCards;
    if ((board_hand & hero_hole_cards) != 0) return error.BoardConflictsWithHeroHoleCards;

    const cards_needed = 5 - @as(u8, @intCast(board.len));
    const used_mask = hero_hole_cards | board_hand;

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

    const BATCH_SIZE = 16;
    const num_batches = simulations / BATCH_SIZE;
    const remainder = simulations % BATCH_SIZE;

    var batch_idx: u32 = 0;
    while (batch_idx < num_batches) : (batch_idx += 1) {
        var hero_batch: @Vector(BATCH_SIZE, u64) = undefined;
        var villain_batch: @Vector(BATCH_SIZE, u64) = undefined;

        inline for (0..BATCH_SIZE) |i| {
            var sampled: u64 = 0;
            var cards_sampled: u8 = 0;

            while (cards_sampled < 2) {
                const idx = rng.uintLessThan(u8, available_count);
                const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
                if ((sampled & card_bit) == 0) {
                    sampled |= card_bit;
                    cards_sampled += 1;
                }
            }
            const villain_hole = sampled;

            var board_sampled: u64 = 0;
            cards_sampled = 0;
            while (cards_sampled < cards_needed) {
                const idx = rng.uintLessThan(u8, available_count);
                const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
                if ((board_sampled & card_bit) == 0 and (villain_hole & card_bit) == 0) {
                    board_sampled |= card_bit;
                    cards_sampled += 1;
                }
            }

            const complete_board = board_hand | board_sampled;
            hero_batch[i] = hero_hole_cards | complete_board;
            villain_batch[i] = villain_hole | complete_board;
        }

        const hero_ranks = evaluator.evaluateBatch(BATCH_SIZE, hero_batch);
        const villain_ranks = evaluator.evaluateBatch(BATCH_SIZE, villain_batch);

        inline for (0..BATCH_SIZE) |i| {
            if (hero_ranks[i] < villain_ranks[i]) {
                wins += 1;
            } else if (hero_ranks[i] == villain_ranks[i]) {
                ties += 1;
            }
        }
    }

    for (0..remainder) |_| {
        var sampled: u64 = 0;
        var cards_sampled: u8 = 0;

        while (cards_sampled < 2) {
            const idx = rng.uintLessThan(u8, available_count);
            const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
            if ((sampled & card_bit) == 0) {
                sampled |= card_bit;
                cards_sampled += 1;
            }
        }
        const villain_hole = sampled;

        var board_sampled: u64 = 0;
        cards_sampled = 0;
        while (cards_sampled < cards_needed) {
            const idx = rng.uintLessThan(u8, available_count);
            const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
            if ((board_sampled & card_bit) == 0 and (villain_hole & card_bit) == 0) {
                board_sampled |= card_bit;
                cards_sampled += 1;
            }
        }

        const complete_board = board_hand | board_sampled;
        const hero_hand = hero_hole_cards | complete_board;
        const villain_hand = villain_hole | complete_board;

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

    // Precompute exclusion mask (board + all hole cards)
    var excluded_mask: Hand = board_hand;
    for (hands) |hole_cards| {
        excluded_mask |= hole_cards;
    }

    var sampler = deck.DeckSampler.init();

    for (0..simulations) |_| {
        sampler.resetWithMask(excluded_mask);
        const board_completion = sampler.drawMask(rng, cards_needed);
        const complete_board = board_hand | board_completion;
        const ctx = evaluator.initBoardContext(complete_board);
        const showdown = evaluator.evaluateShowdownMultiway(&ctx, hands);

        if (showdown.tie_count == 1) {
            const winner_index: usize = @intCast(@ctz(showdown.winner_mask));
            results[winner_index].wins += 1;
        } else {
            var mask = showdown.winner_mask;
            while (mask != 0) {
                const bit_index: usize = @intCast(@ctz(mask));
                results[bit_index].ties += 1;
                mask &= mask - 1;
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

/// Sample remaining cards for equity calculations (unified implementation)
/// Supports both head-to-head and multiway scenarios
fn sampleRemainingCards(hands: []const Hand, board: Hand, num_cards: u8, rng: std.Random) Hand {
    if (num_cards == 0) return 0;

    var exclude_mask: Hand = board;
    for (hands) |hole_cards| {
        exclude_mask |= hole_cards;
    }

    var sampler = deck.DeckSampler.init();
    sampler.removeMask(exclude_mask);
    return sampler.drawMask(rng, num_cards);
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

/// Calculate binomial coefficient C(n, k)
inline fn binomial(n: u32, k: u32) u32 {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;
    if (k == 1) return n;

    const k_use = if (k > n - k) n - k else k;
    var result: u32 = 1;
    var i: u32 = 0;
    while (i < k_use) : (i += 1) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

// === EXPERIMENT 26: GREEDY BATCH CASCADE FOR EXACT EQUITY ===

/// Find largest power of 2 less than or equal to n
inline fn floorPow2(n: usize) usize {
    var p: usize = 1;
    while ((p << 1) <= n) p <<= 1;
    return p;
}

/// Collect remaining single cards (rivers) into stack buffer
/// Returns count of remaining cards
fn collectRemainingSingleCards(out: *[52]u64, hero: Hand, vill: Hand, board: Hand) u8 {
    const used: u64 = hero | vill | board;
    var count: u8 = 0;
    inline for (0..52) |i| {
        const bit = @as(u64, 1) << @intCast(i);
        if ((used & bit) == 0) {
            out[count] = bit;
            count += 1;
        }
    }
    return count;
}

/// Evaluate a compile-time sized chunk of river cards using SIMD batching
inline fn evalChunk(
    comptime N: usize,
    comptime track_categories: bool,
    rivers: []const u64,
    offset: usize,
    hero: Hand,
    vill: Hand,
    board: Hand,
    wins: *u32,
    ties: *u32,
    hand1_categories: *HandCategories,
    hand2_categories: *HandCategories,
) void {
    var hero_batch: @Vector(N, u64) = undefined;
    var vill_batch: @Vector(N, u64) = undefined;

    inline for (0..N) |i| {
        const complete_board = board | rivers[offset + i];
        hero_batch[i] = hero | complete_board;
        vill_batch[i] = vill | complete_board;
    }

    const hero_ranks = evaluator.evaluateBatch(N, hero_batch);
    const vill_ranks = evaluator.evaluateBatch(N, vill_batch);

    inline for (0..N) |i| {
        if (track_categories) {
            hand1_categories.addHand(evaluator.getHandCategory(hero_ranks[i]));
            hand2_categories.addHand(evaluator.getHandCategory(vill_ranks[i]));
        }

        if (hero_ranks[i] < vill_ranks[i]) {
            wins.* += 1;
        } else if (hero_ranks[i] == vill_ranks[i]) {
            ties.* += 1;
        }
    }
}

/// Exact equity for turn→river using greedy batch cascade (zero SIMD waste)
/// Processes 44 rivers with optimal batch sizes: 32+8+4 = 44 lanes
fn exactTurnStreaming(
    comptime track_categories: bool,
    hero: Hand,
    vill: Hand,
    board: Hand,
) EquityResult {
    var rivers: [52]u64 = undefined;
    const total_u8 = collectRemainingSingleCards(&rivers, hero, vill, board);
    const total: usize = @intCast(total_u8);

    var wins: u32 = 0;
    var ties: u32 = 0;
    var hand1_cat_storage = HandCategories{};
    var hand2_cat_storage = HandCategories{};

    var offset: usize = 0;
    var remaining: usize = total;

    // Greedy cascade: process in decreasing power-of-2 chunks
    while (remaining > 0) {
        const chunk = floorPow2(remaining);
        switch (chunk) {
            32 => evalChunk(32, track_categories, rivers[0..total], offset, hero, vill, board, &wins, &ties, &hand1_cat_storage, &hand2_cat_storage),
            16 => evalChunk(16, track_categories, rivers[0..total], offset, hero, vill, board, &wins, &ties, &hand1_cat_storage, &hand2_cat_storage),
            8 => evalChunk(8, track_categories, rivers[0..total], offset, hero, vill, board, &wins, &ties, &hand1_cat_storage, &hand2_cat_storage),
            4 => evalChunk(4, track_categories, rivers[0..total], offset, hero, vill, board, &wins, &ties, &hand1_cat_storage, &hand2_cat_storage),
            2 => evalChunk(2, track_categories, rivers[0..total], offset, hero, vill, board, &wins, &ties, &hand1_cat_storage, &hand2_cat_storage),
            1 => {
                const complete_board = board | rivers[offset];
                const hr = evaluator.evaluateHand(hero | complete_board);
                const vr = evaluator.evaluateHand(vill | complete_board);

                if (track_categories) {
                    hand1_cat_storage.addHand(evaluator.getHandCategory(hr));
                    hand2_cat_storage.addHand(evaluator.getHandCategory(vr));
                }

                if (hr < vr) {
                    wins += 1;
                } else if (hr == vr) {
                    ties += 1;
                }
            },
            else => unreachable,
        }
        offset += chunk;
        remaining -= chunk;
    }

    return .{
        .wins = wins,
        .ties = ties,
        .total_simulations = @intCast(total),
        .hand1_categories = if (track_categories) hand1_cat_storage else null,
        .hand2_categories = if (track_categories) hand2_cat_storage else null,
        .method = .exact,
    };
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

    // Pre-calculate total combinations and pre-allocate
    const total_combos = binomial(remaining_count, num_cards);
    const combinations = try allocator.alloc(Hand, total_combos);
    errdefer allocator.free(combinations);

    // Generate combinations iteratively
    var indices: [5]u8 = undefined;
    for (0..num_cards) |i| {
        indices[i] = @intCast(i);
    }

    var combo_idx: usize = 0;
    while (true) {
        // Generate current combination
        var combo: u64 = 0;
        for (0..num_cards) |i| {
            combo |= remaining_cards[indices[i]];
        }
        combinations[combo_idx] = combo;
        combo_idx += 1;

        // Find next combination
        var i: i8 = @intCast(num_cards - 1);
        while (i >= 0) : (i -= 1) {
            const idx = @as(usize, @intCast(i));
            const idx_u8 = @as(u8, @intCast(idx));
            if (indices[idx] < remaining_count - (num_cards - idx_u8)) {
                indices[idx] += 1;
                for (idx + 1..num_cards) |j| {
                    indices[j] = indices[j - 1] + 1;
                }
                break;
            }
        } else {
            break;
        }
    }

    return combinations;
}

// === THREADED EQUITY CALCULATION ===

// Cache-line padded thread result to prevent false sharing
// Note: Each element in the array should be aligned to cache line boundary
const ThreadResult = struct {
    wins: u32 = 0,
    ties: u32 = 0,
    total_simulations: u32 = 0,

    // Pad to cache line size (64 bytes on most architectures)
    _padding: [64 - 3 * @sizeOf(u32)]u8 = .{0} ** (64 - 3 * @sizeOf(u32)),

    comptime {
        if (@sizeOf(ThreadResult) != 64) {
            @compileError("ThreadResult size must be 64 bytes for cache line alignment");
        }
    }
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
        const board_completion = sampleRemainingCards(&[_]Hand{ ctx.hero_hole_cards, ctx.villain_hole_cards }, ctx.board, cards_needed, rng);

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

    // Validate board
    const board_hand = try validateBoard(hero_hole_cards, villain_hole_cards, board);

    // Get optimal thread count (but cap at reasonable limit)
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const sims_per_thread = simulations / thread_count;
    const remaining_sims = simulations % thread_count;

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
    // Verify ThreadResult is properly padded to prevent false sharing
    try testing.expect(@sizeOf(ThreadResult) == 64);
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

test "monteCarlo validates board length" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Board with 6 cards (too many)
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
        card.makeCard(.clubs, .two),
        card.makeCard(.spades, .three),
        card.makeCard(.hearts, .four),
    };

    try testing.expectError(error.InvalidBoardLength, monteCarlo(aa, kk, &board, 100, rng, allocator));
}

test "monteCarlo validates board duplicates" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Board with duplicate cards
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.spades, .seven), // Duplicate
        card.makeCard(.diamonds, .nine),
    };

    try testing.expectError(error.DuplicateBoardCards, monteCarlo(aa, kk, &board, 100, rng, allocator));
}

test "monteCarlo validates board conflicts with hole cards" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);
    const kk = card.makeCard(.hearts, .king) | card.makeCard(.spades, .king);

    // Board contains a card from hero's hand
    const board = [_]Hand{
        card.makeCard(.clubs, .ace), // Conflicts with hero
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
    };

    try testing.expectError(error.BoardConflictsWithHeroHoleCards, monteCarlo(aa, kk, &board, 100, rng, allocator));
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

test "EquityResult confidence interval for Monte Carlo" {
    const result = EquityResult{
        .wins = 500,
        .ties = 0,
        .total_simulations = 1000,
        .method = .monte_carlo,
    };

    const ci = result.confidenceInterval().?;

    // 50% equity, should have reasonable confidence bounds
    try testing.expect(ci.lower >= 0.0);
    try testing.expect(ci.upper <= 1.0);
    try testing.expect(ci.lower < 0.5);
    try testing.expect(ci.upper > 0.5);
}

test "EquityResult confidence interval returns null for exact" {
    const result = EquityResult{
        .wins = 500,
        .ties = 0,
        .total_simulations = 1000,
        .method = .exact,
    };

    try testing.expect(result.confidenceInterval() == null);
}

test "EquityResult methods handle zero simulations" {
    const result = EquityResult{
        .wins = 0,
        .ties = 0,
        .total_simulations = 0,
    };

    try testing.expect(result.winRate() == 0.0);
    try testing.expect(result.tieRate() == 0.0);
    try testing.expect(result.lossRate() == 0.0);
    try testing.expect(result.equity() == 0.0);
    try testing.expect(result.confidenceInterval() == null);
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

test "exactVsRandom AA should have ~85% equity" {
    // Skip: This test takes ~10 seconds (2B evaluations) - run manually if needed
    return error.SkipZigTest;

    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();

    // const aa = card.makeCard(.clubs, .ace) | card.makeCard(.diamonds, .ace);

    // const result = try exactVsRandom(aa, allocator);

    // // AA vs random should be close to 85.2%
    // try testing.expect(result.equity() > 0.84);
    // try testing.expect(result.equity() < 0.86);

    // // Total simulations should be C(50,2) * C(48,5) = 1,225 * 1,712,304 = 2,097,572,400
    // try testing.expect(result.total_simulations > 2_000_000_000);
}

test "exactVsRandom 22 should have ~50% equity" {
    // Skip: This test takes ~10 seconds (2B evaluations) - run manually if needed
    return error.SkipZigTest;

    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();
    // const allocator = gpa.allocator();

    // const twos = card.makeCard(.clubs, .two) | card.makeCard(.diamonds, .two);

    // const result = try exactVsRandom(twos, allocator);

    // // 22 vs random should be close to 50.3%
    // try testing.expect(result.equity() > 0.49);
    // try testing.expect(result.equity() < 0.52);
}

test "equityVsRandom AA on flop should be ~89%" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.hearts, .ace) | card.makeCard(.spades, .ace);
    const board = [_]Hand{
        card.makeCard(.diamonds, .king),
        card.makeCard(.clubs, .seven),
        card.makeCard(.spades, .two),
    };

    const result = try equityVsRandom(aa, &board, 50000, rng, allocator);

    try testing.expect(result.equity() > 0.85);
    try testing.expect(result.equity() < 0.93);
}

test "equityVsRandom preflop AA should be ~85%" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = card.makeCard(.hearts, .ace) | card.makeCard(.spades, .ace);

    const result = try equityVsRandom(aa, &.{}, 100000, rng, allocator);

    try testing.expect(result.equity() > 0.82);
    try testing.expect(result.equity() < 0.88);
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
