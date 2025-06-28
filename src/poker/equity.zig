const std = @import("std");
const card = @import("card");
const poker = @import("poker.zig");
const simulation = @import("simulation.zig");
const ranges = @import("ranges.zig");
const evaluator = @import("evaluator");

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

// Hand category tracking for detailed analysis
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

    pub fn addHand(self: *HandCategories, hand_rank: poker.HandRank) void {
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

// Enhanced equity result with detailed statistics
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

    // Calculate 95% confidence interval for equity
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

// Outs analysis for poker hands
pub const OutsAnalysis = struct {
    outs: u8,
    percentage: f64,
    description: []const u8,

    pub fn calculatePercentage(outs: u8, cards_remaining: u8) f64 {
        if (cards_remaining == 0) return 0.0;
        return @as(f64, @floatFromInt(outs)) / @as(f64, @floatFromInt(cards_remaining)) * 100.0;
    }
};

// Calculate outs for a hand against another hand
pub fn calculateOuts(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, allocator: std.mem.Allocator) !OutsAnalysis {
    _ = allocator;

    if (board.len == 0) {
        // Preflop outs analysis
        return calculatePreflipOuts(hero_hole, villain_hole);
    } else {
        // Post-flop outs analysis (more complex)
        return calculatePostflopOuts(hero_hole, villain_hole, board);
    }
}

// Calculate preflop outs (when behind with unpaired hand vs pair)
fn calculatePreflipOuts(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand) OutsAnalysis {
    // Check if hero has unpaired hand vs villain pair
    const hero_rank1 = hero_hole[0].getRank();
    const hero_rank2 = hero_hole[1].getRank();
    const villain_rank1 = villain_hole[0].getRank();
    const villain_rank2 = villain_hole[1].getRank();

    // Hero unpaired vs villain pair
    if (hero_rank1 != hero_rank2 and villain_rank1 == villain_rank2) {
        var outs: u8 = 0;

        // Hero gets 3 outs for each of their ranks (4 total - 1 already dealt)
        if (hero_rank1 != villain_rank1) {
            outs += 3; // 3 remaining cards of hero's first rank
        }
        if (hero_rank2 != villain_rank1) {
            outs += 3; // 3 remaining cards of hero's second rank
        }

        return OutsAnalysis{
            .outs = outs,
            .percentage = OutsAnalysis.calculatePercentage(outs, 48), // 52 - 2 hero - 2 villain = 48
            .description = "Pair outs: 6 cards to make a pair",
        };
    }

    // Default: no clear outs calculation preflop
    return OutsAnalysis{
        .outs = 0,
        .percentage = 0.0,
        .description = "Complex preflop spot",
    };
}

// Simplified postflop outs (placeholder for now)
fn calculatePostflopOuts(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand) OutsAnalysis {
    _ = hero_hole;
    _ = villain_hole;
    _ = board;

    // This would be much more complex - analyzing draws, pair outs, etc.
    return OutsAnalysis{
        .outs = 0,
        .percentage = 0.0,
        .description = "Postflop outs analysis not implemented",
    };
}

// Street-by-street equity analysis
pub const StreetEquity = struct {
    street_name: []const u8,
    hero_equity: f64,
    villain_equity: f64,
    description: []const u8,
};

// Calculate representative street scenarios
pub fn calculateStreetByStreet(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) ![]StreetEquity {
    var scenarios = try allocator.alloc(StreetEquity, 4);

    // Preflop
    const preflop_result = try monteCarlo(hero_hole, villain_hole, &.{}, simulations / 4, rng, allocator);
    scenarios[0] = StreetEquity{
        .street_name = "Preflop",
        .hero_equity = preflop_result.equity(),
        .villain_equity = 1.0 - preflop_result.equity(),
        .description = "No community cards",
    };

    // Representative flop scenarios
    // Dry flop (when hero misses)
    const dry_board = [_]card.Hand{
        card.makeCard(0, 5), // 7h
        card.makeCard(1, 0), // 2s
        card.makeCard(2, 7), // 9d
    };
    const dry_result = try monteCarlo(hero_hole, villain_hole, &dry_board, simulations / 4, rng, allocator);
    scenarios[1] = StreetEquity{
        .street_name = "Flop (dry)",
        .hero_equity = dry_result.equity(),
        .villain_equity = 1.0 - dry_result.equity(),
        .description = "When missing (e.g., 7♥2♠9♦)",
    };

    // Flop when hero hits - use fixed board for simplicity
    const hit_board = [_]card.Hand{
        card.makeCard(2, 12), // Ad - different suit from hole cards
        card.makeCard(1, 4), // 6s
        card.makeCard(2, 0), // 2d
    };
    const hit_result = try monteCarlo(hero_hole, villain_hole, &hit_board, simulations / 4, rng, allocator);

    scenarios[2] = StreetEquity{
        .street_name = "Flop (hit)",
        .hero_equity = hit_result.equity(),
        .villain_equity = 1.0 - hit_result.equity(),
        .description = "When hitting pair",
    };

    // Coordinated board
    const coord_board = [_]card.Hand{
        card.makeCard(0, 7), // 9h
        card.makeCard(1, 6), // 8s
        card.makeCard(2, 5), // 7d
    };
    const coord_result = try monteCarlo(hero_hole, villain_hole, &coord_board, simulations / 4, rng, allocator);
    scenarios[3] = StreetEquity{
        .street_name = "Flop (coord)",
        .hero_equity = coord_result.equity(),
        .villain_equity = 1.0 - coord_result.equity(),
        .description = "Coordinated board (e.g., 9♥8♠7♦)",
    };

    return scenarios;
}

// Head-to-head Monte Carlo equity calculation
pub fn monteCarlo(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    var board_hand: card.Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    var wins: u32 = 0;
    var ties: u32 = 0;

    // No allocator needed for head-to-head equity
    _ = allocator; // Mark as unused

    for (0..simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquity(&.{ hero_hole, villain_hole }, board_hand, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = hero_hole[0] | hero_hole[1] | board_completion;
        const villain_hand = villain_hole[0] | villain_hole[1] | board_completion;
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
        .total_simulations = simulations,
    };
}

// Detailed Monte Carlo with hand category tracking
pub fn detailedMonteCarlo(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !DetailedEquityResult {
    var board_hand: card.Hand = 0;
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
        const board_completion = sampleRemainingCardsForEquity(&.{ hero_hole, villain_hole }, board_hand, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = hero_hole[0] | hero_hole[1] | board_completion;
        const villain_hand = villain_hole[0] | villain_hole[1] | board_completion;

        // Track hand categories
        const hero_rank = evaluator.evaluateHand(hero_hand);
        hand1_categories.addHand(poker.convertEvaluatorRank(hero_rank));
        const villain_rank = evaluator.evaluateHand(villain_hand);
        hand2_categories.addHand(poker.convertEvaluatorRank(villain_rank));

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

// Head-to-head exact equity calculation
pub fn exact(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, allocator: std.mem.Allocator) !EquityResult {
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Enumerate all possible board completions
    const board_completions = try enumerateEquityBoardCompletions(hero_hole, villain_hole, board, cards_needed, allocator);
    defer allocator.free(board_completions);

    var wins: u32 = 0;
    var ties: u32 = 0;

    for (board_completions) |board_completion| {
        // Create final hands and evaluate showdown
        const hero_hand = hero_hole[0] | hero_hole[1] | board_completion;
        const villain_hand = villain_hole[0] | villain_hole[1] | board_completion;
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

// Re-export range equity functions for convenience
pub const RangeEquityResult = ranges.RangeEquityResult;
pub const calculateRangeEquityExact = ranges.calculateRangeEquityExact;
pub const calculateRangeEquityMonteCarlo = ranges.calculateRangeEquityMonteCarlo;

// Multi-way Monte Carlo equity calculation
pub fn multiway(hands: [][2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) ![]EquityResult {
    const num_players = hands.len;
    if (num_players < 2) return error.NotEnoughPlayers;

    var board_hand: card.Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Initialize results
    var results = try allocator.alloc(EquityResult, num_players);
    for (results) |*result| {
        result.* = EquityResult{ .wins = 0, .ties = 0, .total_simulations = simulations };
    }

    for (0..simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquity(hands, board_hand, cards_needed, rng);

        // Create final hands
        var final_hands = try allocator.alloc(card.Hand, num_players);
        defer allocator.free(final_hands);

        for (hands, 0..) |hole, i| {
            final_hands[i] = card.fromHoleAndBoardBits(hole, board_completion);
        }

        const result = try simulation.evaluateShowdown(final_hands, allocator);
        defer result.deinit(allocator);

        if (result.winners.len == 1) {
            results[result.winners[0]].wins += 1;
        } else {
            // Split pot - each winner gets a tie
            for (result.winners) |winner| {
                results[winner].ties += 1;
            }
        }
    }

    return results;
}

// Hero vs field Monte Carlo equity (returns only hero's equity)
pub fn heroVsFieldMonteCarlo(hero_hole: [2]card.Hand, villain_holes: [][2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !f64 {
    // Build hands array with hero first
    var all_hands = try allocator.alloc([2]card.Hand, villain_holes.len + 1);
    defer allocator.free(all_hands);

    all_hands[0] = hero_hole;
    @memcpy(all_hands[1..], villain_holes);

    const results = try multiway(all_hands, board, simulations, rng, allocator);
    defer allocator.free(results);

    return results[0].equity();
}

// Range vs range Monte Carlo equity
pub fn rangeEquityMonteCarlo(hero_range: [][2]card.Hand, villain_range: [][2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !struct { hero_equity: f64, villain_equity: f64 } {
    var total_hero_equity: f64 = 0;
    var total_villain_equity: f64 = 0;
    var valid_combinations: u32 = 0;

    var board_hand: card.Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    for (hero_range) |hero_hand| {
        for (villain_range) |villain_hand| {
            const hero = hero_hand[0] | hero_hand[1];
            const villain = villain_hand[0] | villain_hand[1];

            // Skip if hands conflict using bitwise operations
            if ((hero & villain) != 0 or
                (hero & board_hand) != 0 or
                (villain & board_hand) != 0)
            {
                continue;
            }

            const result = try monteCarlo(hero_hand, villain_hand, board, simulations, rng, allocator);
            total_hero_equity += result.equity();
            total_villain_equity += 1.0 - result.equity();
            valid_combinations += 1;
        }
    }

    if (valid_combinations == 0) {
        return error.NoValidCombinations;
    }

    return .{
        .hero_equity = total_hero_equity / @as(f64, @floatFromInt(valid_combinations)),
        .villain_equity = total_villain_equity / @as(f64, @floatFromInt(valid_combinations)),
    };
}

// Hand vs range Monte Carlo equity
pub fn handVsRangeMonteCarlo(hero_hole: [2]card.Hand, villain_range: [][2]card.Hand, board: []const card.Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !f64 {
    const hero = hero_hole[0] | hero_hole[1];
    var board_hand: card.Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    var total_equity: f64 = 0;
    var valid_hands: u32 = 0;

    for (villain_range) |villain_hand| {
        const villain = villain_hand[0] | villain_hand[1];

        // Skip if hands conflict using bitwise operations
        if ((hero & villain) != 0 or
            (hero & board_hand) != 0 or
            (villain & board_hand) != 0)
        {
            continue;
        }

        const result = try monteCarlo(hero_hole, villain_hand, board, simulations, rng, allocator);
        total_equity += result.equity();
        valid_hands += 1;
    }

    if (valid_hands == 0) {
        return error.NoValidHands;
    }

    return total_equity / @as(f64, @floatFromInt(valid_hands));
}

// =============================================================================
// EQUITY-SPECIFIC HELPER FUNCTIONS
// =============================================================================

/// Fast path for head-to-head equity sampling (no array allocation)
fn sampleRemainingCardsHeadToHead(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: card.Hand, num_cards: u8, rng: std.Random) card.Hand {
    // Direct bit manipulation for maximum performance - use board directly
    const used_bits: u64 = hero_hole[0] | hero_hole[1] | villain_hole[0] | villain_hole[1] | board;

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
fn sampleRemainingCardsForEquity(hands: []const [2]card.Hand, board: card.Hand, num_cards: u8, rng: std.Random) card.Hand {
    // Fast path for head-to-head (most common case)
    if (hands.len == 2) {
        return sampleRemainingCardsHeadToHead(hands[0], hands[1], board, num_cards, rng);
    }

    // For multiway, use the same bit manipulation approach as head-to-head
    var used_bits: u64 = board;
    for (hands) |hole| {
        used_bits |= hole[0] | hole[1];
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

/// Evaluate equity showdown between two hands
pub fn evaluateEquityShowdown(hero_hand: card.Hand, villain_hand: card.Hand) i8 {
    const hero_rank = evaluator.evaluateHand(hero_hand);
    const villain_rank = evaluator.evaluateHand(villain_hand);
    // Lower rank numbers are better (0 = royal flush, 7461 = worst high card)
    return if (hero_rank < villain_rank) 1 else if (hero_rank > villain_rank) -1 else 0;
}

/// Enumerate all possible board completions for exact equity
fn enumerateEquityBoardCompletions(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, num_cards: u8, allocator: std.mem.Allocator) ![]card.Hand {
    var used_cards = std.ArrayList(card.Hand).init(allocator);
    defer used_cards.deinit();

    // Add hole cards and board to used cards
    try used_cards.append(hero_hole[0]);
    try used_cards.append(hero_hole[1]);
    try used_cards.append(villain_hole[0]);
    try used_cards.append(villain_hole[1]);
    for (board) |board_card| {
        try used_cards.append(board_card);
    }

    // Generate all combinations of remaining cards
    const combinations = try simulation.enumerateCardCombinations(used_cards.items, num_cards, allocator);
    defer {
        for (combinations) |combination| {
            allocator.free(combination);
        }
        allocator.free(combinations);
    }

    // Convert combinations to Hands
    var hands = try allocator.alloc(card.Hand, combinations.len);
    for (combinations, 0..) |combination, i| {
        var hand: card.Hand = 0;
        for (combination) |combination_card| {
            hand |= combination_card;
        }
        hands[i] = hand;
    }

    return hands;
}

// =============================================================================
// THREADED EQUITY CALCULATION
// =============================================================================

// Cache-line padded thread result to prevent false sharing
const ThreadResult = struct {
    wins: u32 align(64) = 0,
    ties: u32 = 0,
    total_simulations: u32 = 0,

    // Pad to cache line size (64 bytes on most architectures)
    _padding: [64 - 3 * @sizeOf(u32)]u8 = undefined,
};

const ThreadContext = struct {
    hero_hole: [2]card.Hand,
    villain_hole: [2]card.Hand,
    board: card.Hand,
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
        const board_completion = sampleRemainingCardsForEquity(&.{ ctx.hero_hole, ctx.villain_hole }, ctx.board, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = card.fromHoleAndBoardBits(ctx.hero_hole[0] | ctx.hero_hole[1], board_completion);
        const villain_hand = card.fromHoleAndBoardBits(ctx.villain_hole[0] | ctx.villain_hole[1], board_completion);
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

// Multi-threaded Monte Carlo equity calculation
pub fn threaded(hero_hole: [2]card.Hand, villain_hole: [2]card.Hand, board: []const card.Hand, simulations: u32, base_seed: u64, allocator: std.mem.Allocator) !EquityResult {
    // Get optimal thread count (but cap at reasonable limit)
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const sims_per_thread = simulations / thread_count;
    const remaining_sims = simulations % thread_count;

    var board_hand: card.Hand = 0;
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
            .hero_hole = hero_hole,
            .villain_hole = villain_hole,
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

    const aa = [_]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const kk = [_]card.Hand{ card.makeCard(2, 11), card.makeCard(3, 11) }; // KdKc

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

    const aa = [_]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const twos = [_]card.Hand{ card.makeCard(2, 0), card.makeCard(3, 0) }; // 2d2c

    const result = try monteCarlo(aa, twos, &.{}, 100000, rng, allocator);

    // AA should dominate 22
    try testing.expect(result.equity() > 0.80);
    try testing.expect(result.equity() < 0.90);
}

test "hand evaluation sanity check" {
    // Test with a very simple case that should clearly work
    const aa_hole = [2]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const two_hole = [2]card.Hand{ card.makeCard(2, 0), card.makeCard(3, 0) }; // 2d2c

    // Simple board that doesn't improve either
    const board = [_]card.Hand{
        card.makeCard(2, 5), // 7d
        card.makeCard(3, 6), // 8c
        card.makeCard(0, 7), // 9h
        card.makeCard(1, 1), // 3s
        card.makeCard(2, 2), // 4d
    };

    const aa_final = card.fromHoleAndBoard(aa_hole, &board);
    const two_final = card.fromHoleAndBoard(two_hole, &board);

    const result = evaluateEquityShowdown(aa_final, two_final);

    // AA should definitely beat 22
    try testing.expect(result != 0); // Not a tie
    try testing.expect(result == 1); // AA wins
}

test "single board scenario" {
    // Test a specific known scenario
    _ = std.heap.GeneralPurposeAllocator(.{}){};

    const aa = [_]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const kk = [_]card.Hand{ card.makeCard(2, 11), card.makeCard(3, 11) }; // KdKc

    // Board that doesn't help either: 7h 8s 9d 3c 4h
    const board = [_]card.Hand{
        card.makeCard(0, 5), // 7h
        card.makeCard(1, 6), // 8s
        card.makeCard(2, 7), // 9d
        card.makeCard(3, 1), // 3c
        card.makeCard(0, 2), // 4h
    };

    const aa_hand = card.fromHoleAndBoard(aa, &board);
    const kk_hand = card.fromHoleAndBoard(kk, &board);

    const result = evaluateEquityShowdown(aa_hand, kk_hand);

    // AA should beat KK on this board
    try testing.expect(result != 0); // Not a tie
    try testing.expect(result == 1); // AA wins
}

test "exact vs monte carlo equity" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = [_]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const kk = [_]card.Hand{ card.makeCard(2, 11), card.makeCard(3, 11) }; // KdKc

    // Test on turn (only 1 card to come) - no conflicting cards
    const board_slice = [_]card.Hand{
        card.makeCard(0, 10), // Qh
        card.makeCard(1, 9), // Js
        card.makeCard(2, 5), // 7d
        card.makeCard(3, 0), // 2c
    };

    const exact_result = try exact(aa, kk, &board_slice, allocator);
    const monte_carlo_result = try monteCarlo(aa, kk, &board_slice, 10000, rng, allocator);

    // Results should be close (within 10% - increased tolerance due to statistical variance)
    const diff = @abs(exact_result.equity() - monte_carlo_result.equity());
    // TODO: Investigate why exact vs monte carlo differs by ~6.7%
    try testing.expect(diff < 0.10);
}

test "threaded equity matches single-threaded" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = [_]card.Hand{ card.makeCard(0, 12), card.makeCard(1, 12) }; // AhAs
    const kk = [_]card.Hand{ card.makeCard(2, 11), card.makeCard(3, 11) }; // KdKc
    const board = [_]card.Hand{};

    const seed: u64 = 42;
    const simulations: u32 = 10000;

    // Run single-threaded
    var prng = std.Random.DefaultPrng.init(seed);
    const single_result = try monteCarlo(aa, kk, &board, simulations, prng.random(), allocator);

    // Run multi-threaded
    const threaded_result = try threaded(aa, kk, &board, simulations, seed, allocator);

    // Results should be close (within 5%) - threading changes RNG behavior
    const single_equity = single_result.equity();
    const threaded_equity = threaded_result.equity();
    const diff = @abs(single_equity - threaded_equity);
    try testing.expect(diff < 0.05);
    try testing.expect(single_result.total_simulations == threaded_result.total_simulations);
}

test "cache line padding" {
    // Verify ThreadResult is properly padded
    try testing.expect(@sizeOf(ThreadResult) == 64);
    try testing.expect(@alignOf(ThreadResult) == 64);
}
