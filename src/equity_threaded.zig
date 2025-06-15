const std = @import("std");
const poker = @import("poker.zig");
const simulation = @import("simulation.zig");
const equity = @import("equity.zig");

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
    board: []const poker.Card,
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

    const hero_bits = simulation.cardsToHoleBits(ctx.hero_hole);
    const villain_bits = simulation.cardsToHoleBits(ctx.villain_hole);
    const board_bits = simulation.boardToBits(ctx.board);
    const used_cards = hero_bits | villain_bits | board_bits;

    const cards_needed = 5 - @as(u8, @intCast(ctx.board.len));

    var wins: u32 = 0;
    var ties: u32 = 0;

    // Run simulations assigned to this thread
    for (0..ctx.simulations) |_| {
        // Sample remaining board cards
        const remaining_board = simulation.sampleRemainingCards(used_cards, cards_needed, rng);
        const final_board = board_bits | remaining_board;

        // Create final hands
        const hero_hand = simulation.combineCards(hero_bits, final_board);
        const villain_hand = simulation.combineCards(villain_bits, final_board);

        // Use fast path for 2-player (zero allocation)
        const result = simulation.evaluateShowdownHeadToHead(hero_hand, villain_hand);

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
pub fn equityMonteCarloThreaded(hero_hole: [2]poker.Card, villain_hole: [2]poker.Card, board: []const poker.Card, simulations: u32, base_seed: u64, allocator: std.mem.Allocator) !equity.EquityResult {
    // Get optimal thread count (but cap at reasonable limit)
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const sims_per_thread = simulations / thread_count;
    const remaining_sims = simulations % thread_count;

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
            .board = board,
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

    return equity.EquityResult{
        .wins = total_wins,
        .ties = total_ties,
        .total_simulations = total_sims,
    };
}

// Tests
const testing = std.testing;

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
    const single_result = try equity.equityMonteCarlo(aa, kk, &board, simulations, prng.random(), allocator);

    // Run multi-threaded
    const threaded_result = try equityMonteCarloThreaded(aa, kk, &board, simulations, seed, allocator);

    // Results should be deterministic and identical
    try testing.expect(single_result.wins == threaded_result.wins);
    try testing.expect(single_result.ties == threaded_result.ties);
    try testing.expect(single_result.total_simulations == threaded_result.total_simulations);
}

test "cache line padding" {
    // Verify ThreadResult is properly padded
    try testing.expect(@sizeOf(ThreadResult) == 64);
    try testing.expect(@alignOf(ThreadResult) == 64);
}
