const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");

/// Hot path profiling tool for pokerbot workloads
/// Usage: profile_main --scenario=<name> [--iterations=N]
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var scenario: ?[]const u8 = null;
    var iterations: u32 = 10_000_000;

    // Parse flags
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--scenario=")) {
            scenario = arg["--scenario=".len..];
        } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
            iterations = try std.fmt.parseInt(u32, arg["--iterations=".len..], 10);
        }
    }

    if (scenario == null) {
        std.debug.print("Usage: {s} --scenario=<name> [--iterations=N]\n", .{args[0]});
        std.debug.print("\nAvailable scenarios:\n", .{});
        std.debug.print("  init-board   - Profile initBoardContext (board setup)\n", .{});
        std.debug.print("  showdown     - Profile evaluateShowdownWithContext (heads-up comparison)\n", .{});
        std.debug.print("  multiway     - Profile evaluateHand (6-max equity calculation)\n", .{});
        std.debug.print("\nDefault iterations: 10,000,000\n", .{});
        return;
    }

    if (std.mem.eql(u8, scenario.?, "init-board")) {
        try profileInitBoardContext(iterations);
    } else if (std.mem.eql(u8, scenario.?, "showdown")) {
        try profileShowdown(iterations);
    } else if (std.mem.eql(u8, scenario.?, "multiway")) {
        try profileMultiway(iterations);
    } else {
        std.debug.print("Unknown scenario: {s}\n", .{scenario.?});
        std.debug.print("Available: init-board, showdown, multiway\n", .{});
        return error.InvalidScenario;
    }
}

// Profile initBoardContext - called per sample in blueprint training
fn profileInitBoardContext(iterations: u32) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Generate random boards (3-5 cards each)
    const boards = try generateBoards(1000, rng);

    std.debug.print("Profiling initBoardContext ({} iterations)...\n", .{iterations});

    const start = std.time.nanoTimestamp();
    var checksum: u64 = 0;

    for (0..iterations) |i| {
        const board = boards[i % boards.len];
        const ctx = evaluator.initBoardContext(board);

        // Prevent optimization
        checksum +%= ctx.suit_counts[0];
        checksum +%= ctx.rank_counts[0];
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const ns_per_call = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Time per initBoardContext: {d:.2} ns\n", .{ns_per_call});
    std.debug.print("Calls/sec: {d:.2}M\n", .{1000.0 / ns_per_call});
    std.debug.print("Checksum: {}\n", .{checksum});
}

// Profile evaluateShowdownWithContext - hot loop in payoff sampling
fn profileShowdown(iterations: u32) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Generate realistic scenario: 1 board, many hero/villain pairs
    const board = generateBoard(5, rng);
    const ctx = evaluator.initBoardContext(board);

    const pairs = try generateHolePairs(1000, board, rng);

    std.debug.print("Profiling evaluateShowdownWithContext ({} iterations)...\n", .{iterations});

    const start = std.time.nanoTimestamp();
    var checksum: i64 = 0;

    for (0..iterations) |i| {
        const pair = pairs[i % pairs.len];
        const result = evaluator.evaluateShowdownWithContext(&ctx, pair.hero, pair.villain);
        checksum += result;
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const ns_per_eval = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Time per showdown: {d:.2} ns\n", .{ns_per_eval});
    std.debug.print("Showdowns/sec: {d:.2}M\n", .{1000.0 / ns_per_eval});
    std.debug.print("Checksum: {}\n", .{checksum});
}

// Profile evaluateHand - multiway equity for 6-max
fn profileMultiway(iterations: u32) !void {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Generate 6-max scenario: board + 6 hole card pairs
    const hands = try generate6MaxHands(1000, rng);

    std.debug.print("Profiling evaluateHand for 6-max ({} iterations)...\n", .{iterations});

    const start = std.time.nanoTimestamp();
    var checksum: u64 = 0;

    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const rank = evaluator.evaluateHand(hand);
        checksum +%= rank;
    }

    const elapsed = std.time.nanoTimestamp() - start;
    const ns_per_hand = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));

    std.debug.print("Time per evaluateHand: {d:.2} ns\n", .{ns_per_hand});
    std.debug.print("Hands/sec: {d:.2}M\n", .{1000.0 / ns_per_hand});
    std.debug.print("Checksum: {}\n", .{checksum});
}

// Helper: Generate random boards
fn generateBoards(count: usize, rng: std.Random) ![1000]card.Hand {
    var boards: [1000]card.Hand = undefined;

    for (0..count) |i| {
        const num_cards = 3 + (i % 3); // 3, 4, or 5 cards
        boards[i] = generateBoard(num_cards, rng);
    }

    return boards;
}

fn generateBoard(num_cards: usize, rng: std.Random) card.Hand {
    var board: card.Hand = 0;
    var cards_dealt: usize = 0;

    while (cards_dealt < num_cards) {
        const suit: card.Suit = @enumFromInt(rng.intRangeAtMost(u8, 0, 3));
        const rank: card.Rank = @enumFromInt(rng.intRangeAtMost(u8, 0, 12));
        const card_bits = card.makeCard(suit, rank);

        if ((board & card_bits) == 0) {
            board |= card_bits;
            cards_dealt += 1;
        }
    }

    return board;
}

// Helper: Generate hero/villain hole pairs
const HolePair = struct {
    hero: card.Hand,
    villain: card.Hand,
};

fn generateHolePairs(count: usize, board: card.Hand, rng: std.Random) ![1000]HolePair {
    var pairs: [1000]HolePair = undefined;

    for (0..count) |i| {
        var used = board;
        const hero = generateHole(&used, rng);
        const villain = generateHole(&used, rng);
        pairs[i] = .{ .hero = hero, .villain = villain };
    }

    return pairs;
}

fn generateHole(used: *card.Hand, rng: std.Random) card.Hand {
    var hole: card.Hand = 0;
    var cards_dealt: usize = 0;

    while (cards_dealt < 2) {
        const suit: card.Suit = @enumFromInt(rng.intRangeAtMost(u8, 0, 3));
        const rank: card.Rank = @enumFromInt(rng.intRangeAtMost(u8, 0, 12));
        const card_bits = card.makeCard(suit, rank);

        if ((used.* & card_bits) == 0) {
            hole |= card_bits;
            used.* |= card_bits;
            cards_dealt += 1;
        }
    }

    return hole;
}

// Helper: Generate 6-max full hands (board + hole)
fn generate6MaxHands(count: usize, rng: std.Random) ![1000]card.Hand {
    var hands: [1000]card.Hand = undefined;
    var rand = rng;

    for (0..count) |i| {
        hands[i] = evaluator.generateRandomHand(&rand);
    }

    return hands;
}
