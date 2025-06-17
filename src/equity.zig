const std = @import("std");
const poker = @import("poker.zig");
const simulation = @import("simulation.zig");
const ranges = @import("ranges.zig");

pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,

    pub fn winRate(self: EquityResult) f64 {
        return @as(f64, @floatFromInt(self.wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn equity(self: EquityResult) f64 {
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total_simulations));
    }
};

// Head-to-head Monte Carlo equity calculation
pub fn monteCarlo(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !EquityResult {
    const board_hand = poker.Hand.fromBoard(board);
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    var wins: u32 = 0;
    var ties: u32 = 0;

    // No allocator needed for head-to-head equity
    _ = allocator; // Mark as unused

    for (0..simulations) |_| {
        // Sample remaining board cards
        const board_completion = sampleRemainingCardsForEquity(&.{ hero_hole, villain_hole }, board_hand, cards_needed, rng);

        // Create final hands and evaluate showdown
        const hero_hand = poker.Hand.fromHoleAndBoardBits(hero_hole, board_completion.bits);
        const villain_hand = poker.Hand.fromHoleAndBoardBits(villain_hole, board_completion.bits);
        const result = evaluateEquityShowdown(hero_hand, villain_hand);

        if (!result.tie) {
            if (result.winner == 0) {
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

// Head-to-head exact equity calculation
pub fn exact(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, allocator: std.mem.Allocator) !EquityResult {
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    // Enumerate all possible board completions
    const board_completions = try enumerateEquityBoardCompletions(hero_hole, villain_hole, board, cards_needed, allocator);
    defer allocator.free(board_completions);

    var wins: u32 = 0;
    var ties: u32 = 0;

    for (board_completions) |board_completion| {
        // Create final hands and evaluate showdown
        const hero_hand = poker.Hand.fromHoleAndBoard(hero_hole, board);
        const villain_hand = poker.Hand.fromHoleAndBoard(villain_hole, board);
        const final_hero = hero_hand.combineWith(board_completion);
        const final_villain = villain_hand.combineWith(board_completion);
        const result = evaluateEquityShowdown(final_hero, final_villain);

        if (!result.tie) {
            if (result.winner == 0) {
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
pub fn multiway(hands: [][2]poker.Card, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) ![]EquityResult {
    const num_players = hands.len;
    if (num_players < 2) return error.NotEnoughPlayers;

    const board_hand = poker.Hand.fromBoard(board);
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
        var final_hands = try allocator.alloc(poker.Hand, num_players);
        defer allocator.free(final_hands);

        for (hands, 0..) |hole, i| {
            final_hands[i] = poker.Hand.fromHoleAndBoardBits(hole, board_completion.bits);
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
pub fn heroVsFieldMonteCarlo(hero_hole: [2]poker.Card, villain_holes: [][2]poker.Card, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !f64 {
    // Build hands array with hero first
    var all_hands = try allocator.alloc([2]poker.Card, villain_holes.len + 1);
    defer allocator.free(all_hands);

    all_hands[0] = hero_hole;
    @memcpy(all_hands[1..], villain_holes);

    const results = try multiway(all_hands, board, simulations, rng, allocator);
    defer allocator.free(results);

    return results[0].equity();
}

// Range vs range Monte Carlo equity
pub fn rangeEquityMonteCarlo(hero_range: [][2]poker.Card, villain_range: [][2]poker.Card, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !struct { hero_equity: f64, villain_equity: f64 } {
    var total_hero_equity: f64 = 0;
    var total_villain_equity: f64 = 0;
    var valid_combinations: u32 = 0;

    const board_hand = poker.Hand.fromBoard(board);

    for (hero_range) |hero_hand| {
        for (villain_range) |villain_hand| {
            const hero = poker.Hand{ .bits = hero_hand[0].bits | hero_hand[1].bits };
            const villain = poker.Hand{ .bits = villain_hand[0].bits | villain_hand[1].bits };

            // Skip if hands conflict using Hand methods
            if (hero.hasConflictWith(villain) or
                hero.hasConflictWith(board_hand) or
                villain.hasConflictWith(board_hand))
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
pub fn handVsRangeMonteCarlo(hero_hole: [2]poker.Card, villain_range: [][2]poker.Card, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !f64 {
    const hero = poker.Hand{ .bits = hero_hole[0].bits | hero_hole[1].bits };
    const board_hand = poker.Hand.fromBoard(board);

    var total_equity: f64 = 0;
    var valid_hands: u32 = 0;

    for (villain_range) |villain_hand| {
        const villain = poker.Hand{ .bits = villain_hand[0].bits | villain_hand[1].bits };

        // Skip if hands conflict using Hand methods
        if (hero.hasConflictWith(villain) or
            hero.hasConflictWith(board_hand) or
            villain.hasConflictWith(board_hand))
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
fn sampleRemainingCardsHeadToHead(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: poker.Hand, num_cards: u8, rng: std.Random) poker.Hand {
    // Direct bit manipulation for maximum performance - use board.bits directly
    const used_bits: u64 = hero_hole[0].bits | hero_hole[1].bits | villain_hole[0].bits | villain_hole[1].bits | board.bits;

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

    return poker.Hand{ .bits = sampled_bits };
}

/// Sample remaining cards for equity calculations (performance optimized)
fn sampleRemainingCardsForEquity(hands: []const [2]poker.Card, board: poker.Hand, num_cards: u8, rng: std.Random) poker.Hand {
    // Fast path for head-to-head (most common case)
    if (hands.len == 2) {
        return sampleRemainingCardsHeadToHead(hands[0], hands[1], board, num_cards, rng);
    }

    // For multiway, use the same bit manipulation approach as head-to-head
    var used_bits: u64 = board.bits;
    for (hands) |hole| {
        used_bits |= hole[0].bits | hole[1].bits;
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

    return poker.Hand{ .bits = sampled_bits };
}

/// Evaluate equity showdown between two hands
pub fn evaluateEquityShowdown(hero_hand: poker.Hand, villain_hand: poker.Hand) poker.ShowdownResult {
    return hero_hand.compareWith(villain_hand);
}

/// Enumerate all possible board completions for exact equity
fn enumerateEquityBoardCompletions(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, num_cards: u8, allocator: std.mem.Allocator) ![]poker.Hand {
    var used_cards = std.ArrayList(poker.Card).init(allocator);
    defer used_cards.deinit();

    // Add hole cards and board to used cards
    try used_cards.append(hero_hole[0]);
    try used_cards.append(hero_hole[1]);
    try used_cards.append(villain_hole[0]);
    try used_cards.append(villain_hole[1]);
    for (board) |card| {
        try used_cards.append(card);
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
    var hands = try allocator.alloc(poker.Hand, combinations.len);
    for (combinations, 0..) |combination, i| {
        var hand = poker.Hand.init();
        for (combination) |card| {
            hand.addCard(card);
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
    hero_hole: [2]poker.Card,
    villain_hole: [2]poker.Card,
    board: poker.Hand,
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
        const hero_hand = poker.Hand.fromHoleAndBoardBits(ctx.hero_hole, board_completion.bits);
        const villain_hand = poker.Hand.fromHoleAndBoardBits(ctx.villain_hole, board_completion.bits);
        const result = evaluateEquityShowdown(hero_hand, villain_hand);

        if (!result.tie) {
            if (result.winner == 0) {
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
pub fn threaded(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, simulations: u32, base_seed: u64, allocator: std.mem.Allocator) !EquityResult {
    // Get optimal thread count (but cap at reasonable limit)
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const sims_per_thread = simulations / thread_count;
    const remaining_sims = simulations % thread_count;

    const board_hand = poker.Hand.fromBoard(board);
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

    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc

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

    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const twos = [_]poker.Card{ poker.Card.init(2, 2), poker.Card.init(2, 3) }; // 2d2c

    const result = try monteCarlo(aa, twos, &.{}, 100000, rng, allocator);

    // AA should dominate 22
    try testing.expect(result.equity() > 0.80);
    try testing.expect(result.equity() < 0.90);
}

test "hand evaluation sanity check" {
    // Test with a very simple case that should clearly work
    const aa_hole = [2]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const two_hole = [2]poker.Card{ poker.Card.init(2, 2), poker.Card.init(2, 3) }; // 2d2c

    // Simple board that doesn't improve either
    const board = [_]poker.Card{
        poker.Card.init(7, 2), // 7d
        poker.Card.init(8, 3), // 8c
        poker.Card.init(9, 0), // 9h
        poker.Card.init(3, 1), // 3s
        poker.Card.init(4, 2), // 4d
    };

    const aa_final = poker.Hand.fromHoleAndBoard(aa_hole, &board);
    const two_final = poker.Hand.fromHoleAndBoard(two_hole, &board);

    const result = evaluateEquityShowdown(aa_final, two_final);

    // AA should definitely beat 22
    try testing.expect(!result.tie);
    try testing.expect(result.winner == 0); // AA is player 0
}

test "single board scenario" {
    // Test a specific known scenario
    _ = std.heap.GeneralPurposeAllocator(.{}){};

    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc

    // Board that doesn't help either: 7h 8s 9d 3c 4h
    const board = [_]poker.Card{
        poker.Card.init(7, 0), // 7h
        poker.Card.init(8, 1), // 8s
        poker.Card.init(9, 2), // 9d
        poker.Card.init(3, 3), // 3c
        poker.Card.init(4, 0), // 4h
    };

    const aa_hand = poker.Hand.fromHoleAndBoard(aa, &board);
    const kk_hand = poker.Hand.fromHoleAndBoard(kk, &board);

    const result = evaluateEquityShowdown(aa_hand, kk_hand);

    // AA should beat KK on this board
    try testing.expect(!result.tie);
    try testing.expect(result.winner == 0);
}

test "exact vs monte carlo equity" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc

    // Test on turn (only 1 card to come) - no conflicting cards
    const board_slice = [_]poker.Card{
        poker.Card.init(12, 0), // Qh
        poker.Card.init(11, 1), // Js
        poker.Card.init(7, 2), // 7d
        poker.Card.init(2, 3), // 2c
    };

    const exact_result = try exact(aa, kk, &board_slice, allocator);
    const monte_carlo_result = try monteCarlo(aa, kk, &board_slice, 10000, rng, allocator);

    // Results should be close (within 5%)
    const diff = @abs(exact_result.equity() - monte_carlo_result.equity());
    try testing.expect(diff < 0.05);
}

test "threaded equity matches single-threaded" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc
    const board = [_]poker.Card{};

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
