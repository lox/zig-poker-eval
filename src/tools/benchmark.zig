const std = @import("std");
const poker = @import("poker");

pub const BenchmarkOptions = struct {
    iterations: u32 = 100000,
    warmup: bool = true,
    measure_overhead: bool = true,
    multiple_runs: bool = true,
    show_comparison: bool = true,
    verbose: bool = false,
};

pub const BenchmarkResult = struct {
    batch_ns_per_hand: f64,
    single_ns_per_hand: f64,
    simd_speedup: f64,
    hands_per_second: u64,
    coefficient_variation: f64,
    overhead_ns: f64,
    total_hands: u64,
};

pub const ShowdownBenchmarkResult = struct {
    iterations: u32,
    context_ns_per_eval: f64,
    batch_ns_per_eval: f64,
    speedup: f64,
};

// Helper functions for rigorous benchmarking
const BATCH_SIZE = 32;

const ShowdownCase = struct {
    ctx: poker.BoardContext,
    hero_hole: u64,
    villain_hole: u64,
};

// Helper to create batches from hand arrays
fn createBatch(hands: []const u64, start_idx: usize) @Vector(BATCH_SIZE, u64) {
    var batch_hands: [BATCH_SIZE]u64 = undefined;
    for (0..BATCH_SIZE) |i| {
        batch_hands[i] = hands[(start_idx + i) % hands.len];
    }
    return @as(@Vector(BATCH_SIZE, u64), batch_hands);
}

fn warmupCaches(test_hands: []const u64) void {
    // Touch lookup tables by performing lookups - access through public evaluator API
    var prng = std.Random.DefaultPrng.init(123);
    var rng = prng.random();

    // Warm up by evaluating some random hands
    for (0..1024) |_| {
        const hand = poker.generateRandomHand(&rng);
        std.mem.doNotOptimizeAway(poker.evaluateHand(hand));
    }

    // Touch first portion of hands by evaluating them in batches
    const warmup_hands = @min(65536, test_hands.len); // 64K hands max
    var i: usize = 0;
    while (i + BATCH_SIZE <= warmup_hands) {
        const batch = createBatch(test_hands, i);
        _ = poker.evaluateBatch(BATCH_SIZE, batch);
        i += BATCH_SIZE;
    }
}

fn clearCaches(use_purge: bool) void {
    if (!use_purge) return;

    var child = std.process.Child.init(&[_][]const u8{ "sudo", "purge" }, std.heap.page_allocator);
    _ = child.spawnAndWait() catch return;
}

fn calculateCV(times: []const f64) f64 {
    var sum: f64 = 0;
    for (times) |time| {
        sum += time;
    }
    const mean = sum / @as(f64, @floatFromInt(times.len));

    var variance: f64 = 0;
    for (times) |time| {
        const diff = time - mean;
        variance += diff * diff;
    }
    variance /= @as(f64, @floatFromInt(times.len));

    const std_dev = @sqrt(variance);
    return std_dev / mean;
}

fn runSingleBenchmark(iterations: u32, test_hands: []const u64) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    const total_hands = iterations * BATCH_SIZE;

    var hand_idx: usize = 0;
    for (0..iterations) |_| {
        // Create batch from consecutive hands
        const batch = createBatch(test_hands, hand_idx);

        const results = poker.evaluateBatch(BATCH_SIZE, batch);
        for (0..BATCH_SIZE) |j| {
            checksum +%= results[j];
        }
        hand_idx += BATCH_SIZE;
    }

    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);

    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(total_hands));
}

fn benchmarkDummyEvaluator(iterations: u32, test_hands: []const u64) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    const total_hands = iterations * BATCH_SIZE;

    var hand_idx: usize = 0;
    for (0..iterations) |_| {
        for (0..BATCH_SIZE) |j| {
            checksum +%= @popCount(test_hands[(hand_idx + j) % test_hands.len]); // Trivial operation
        }
        hand_idx += BATCH_SIZE;
    }

    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);

    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(total_hands));
}

fn benchmarkSingleHand(test_hands: []const u64, count: u32) f64 {
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();

    for (0..count) |i| {
        checksum +%= poker.evaluateHand(test_hands[i % test_hands.len]);
    }

    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);

    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(count));
}

fn drawUniqueCard(rng: std.Random, used: *u64) u64 {
    while (true) {
        const idx = rng.uintLessThan(u6, 52);
        const card_bit = @as(u64, 1) << @intCast(idx);
        if ((used.* & card_bit) == 0) {
            used.* |= card_bit;
            return card_bit;
        }
    }
}

fn generateShowdownCases(allocator: std.mem.Allocator, iterations: u32, rng: std.Random) ![]ShowdownCase {
    const cases = try allocator.alloc(ShowdownCase, iterations);
    errdefer allocator.free(cases);

    var index: usize = 0;
    while (index < cases.len) {
        const remaining = cases.len - index;
        const group_size = @min(BATCH_SIZE, remaining);

        var board_used: u64 = 0;
        var board: u64 = 0;
        while (@popCount(board) < 5) {
            board |= drawUniqueCard(rng, &board_used);
        }
        const ctx = poker.initBoardContext(board);

        var pair: usize = 0;
        while (pair < group_size) : (pair += 1) {
            var used: u64 = board;

            var hero_hole: u64 = 0;
            while (@popCount(hero_hole) < 2) {
                hero_hole |= drawUniqueCard(rng, &used);
            }

            var villain_hole: u64 = 0;
            while (@popCount(villain_hole) < 2) {
                villain_hole |= drawUniqueCard(rng, &used);
            }

            cases[index + pair] = .{
                .ctx = ctx,
                .hero_hole = hero_hole,
                .villain_hole = villain_hole,
            };
        }

        index += group_size;
    }

    return cases;
}

fn timeScalarShowdown(cases: []const ShowdownCase) f64 {
    var checksum: i32 = 0;
    const start = std.time.nanoTimestamp();
    for (cases) |case| {
        checksum += poker.evaluateShowdownWithContext(&case.ctx, case.hero_hole, case.villain_hole);
    }
    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);
    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(cases.len));
}

fn timeBatchedShowdown(cases: []const ShowdownCase) f64 {
    var checksum: i32 = 0;
    const start = std.time.nanoTimestamp();
    var index: usize = 0;
    var results_buffer: [BATCH_SIZE]i8 = undefined;
    var hero_buffer: [BATCH_SIZE]poker.Hand = undefined;
    var villain_buffer: [BATCH_SIZE]poker.Hand = undefined;

    while (index < cases.len) {
        const remaining = cases.len - index;
        const chunk = @min(BATCH_SIZE, remaining);

        inline for (0..BATCH_SIZE) |i| {
            if (i < chunk) {
                const case_ref = cases[index + i];
                std.debug.assert(case_ref.ctx.board == cases[index].ctx.board);
                hero_buffer[i] = case_ref.hero_hole;
                villain_buffer[i] = case_ref.villain_hole;
            }
        }

        poker.evaluateShowdownBatch(&cases[index].ctx, hero_buffer[0..chunk], villain_buffer[0..chunk], results_buffer[0..chunk]);
        for (results_buffer[0..chunk]) |res| {
            checksum += res;
        }

        index += chunk;
    }
    const end = std.time.nanoTimestamp();
    std.mem.doNotOptimizeAway(checksum);
    const total_ns = @as(f64, @floatFromInt(end - start));
    return total_ns / @as(f64, @floatFromInt(cases.len));
}

pub fn runBenchmark(options: BenchmarkOptions, allocator: std.mem.Allocator) !BenchmarkResult {
    // Generate test hands
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const num_test_hands = 1_600_000; // Large pool for cache effects
    const test_hands = try allocator.alloc(u64, num_test_hands);
    defer allocator.free(test_hands);

    for (test_hands) |*hand| {
        hand.* = poker.generateRandomHand(&rng);
    }

    var result = BenchmarkResult{
        .batch_ns_per_hand = 0,
        .single_ns_per_hand = 0,
        .simd_speedup = 0,
        .hands_per_second = 0,
        .coefficient_variation = 0,
        .overhead_ns = 0,
        .total_hands = 0,
    };

    // Cache warmup
    if (options.warmup) {
        warmupCaches(test_hands);
    }

    // Measure overhead
    if (options.measure_overhead) {
        result.overhead_ns = benchmarkDummyEvaluator(options.iterations / 10, test_hands);
    }

    // Batch benchmark
    if (options.multiple_runs) {
        // Multiple runs for statistical analysis
        const NUM_RUNS = 5;
        const use_purge = false; // Set to true if running with sudo
        var times: [NUM_RUNS]f64 = undefined;

        for (0..NUM_RUNS) |run| {
            if (run > 0) {
                clearCaches(use_purge);
                if (use_purge) {
                    std.time.sleep(3_000_000_000); // 3 seconds
                }
            }

            times[run] = runSingleBenchmark(options.iterations, test_hands);
        }

        // Calculate statistics
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        const median = times[NUM_RUNS / 2];
        result.batch_ns_per_hand = median - result.overhead_ns;
        result.coefficient_variation = calculateCV(&times);
    } else {
        // Single run
        const raw_time = runSingleBenchmark(options.iterations, test_hands);
        result.batch_ns_per_hand = raw_time - result.overhead_ns;
        result.coefficient_variation = 0.0;
    }

    // Single-hand benchmark for comparison
    if (options.show_comparison) {
        result.single_ns_per_hand = benchmarkSingleHand(test_hands, 10000);
        result.simd_speedup = result.single_ns_per_hand / result.batch_ns_per_hand;
    }

    result.hands_per_second = @as(u64, @intFromFloat(1_000_000_000.0 / result.batch_ns_per_hand));
    result.total_hands = options.iterations * BATCH_SIZE;

    return result;
}

pub fn validateCorrectness(test_hands: []const u64) !bool {
    var matches: u32 = 0;
    var total: u32 = 0;

    // Validate first 16K hands in batches
    const validation_hands = @min(16000, test_hands.len);
    var i: usize = 0;
    while (i + BATCH_SIZE <= validation_hands) {
        const batch = createBatch(test_hands, i);

        const fast_results = poker.evaluateBatch(BATCH_SIZE, batch);

        for (0..BATCH_SIZE) |j| {
            const slow_result = poker.slow.evaluateHand(test_hands[i + j]);
            const fast_result = fast_results[j];

            if (slow_result == fast_result) {
                matches += 1;
            }
            total += 1;
        }
        i += BATCH_SIZE;
    }

    const accuracy = @as(f64, @floatFromInt(matches)) / @as(f64, @floatFromInt(total));

    // Require 100% accuracy
    if (accuracy < 1.0) {
        return error.AccuracyTooLow;
    }

    return true;
}

pub fn benchmarkShowdown(allocator: std.mem.Allocator, iterations: u32) !ShowdownBenchmarkResult {
    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const cases = try generateShowdownCases(allocator, iterations, rng);
    defer allocator.free(cases);

    // Warm cache by running each path once before timing
    std.mem.doNotOptimizeAway(timeScalarShowdown(cases));
    std.mem.doNotOptimizeAway(timeBatchedShowdown(cases));

    const context_ns = timeScalarShowdown(cases);
    const batch_ns = timeBatchedShowdown(cases);

    return ShowdownBenchmarkResult{
        .iterations = iterations,
        .context_ns_per_eval = context_ns,
        .batch_ns_per_eval = batch_ns,
        .speedup = context_ns / batch_ns,
    };
}

// Test the evaluator with a specific test batch
pub fn testEvaluator() !void {
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    // Generate a batch using the comptime-sized function
    const batch = poker.generateRandomHandBatch(BATCH_SIZE, &rng);

    // Evaluate batch
    const batch_results = poker.evaluateBatch(BATCH_SIZE, batch);

    // Validate against single-hand evaluation
    var matches: u32 = 0;

    for (0..BATCH_SIZE) |i| {
        const hand = batch[i];
        const batch_result = batch_results[i];
        const single_result = poker.slow.evaluateHand(hand);

        if (batch_result == single_result) {
            matches += 1;
        }
    }

    if (matches != BATCH_SIZE) {
        return error.EvaluatorMismatch;
    }
}

// Test a specific hand for debugging
pub fn testSingleHand(hand: u64) struct { slow: u16, fast: u16, match: bool } {
    const slow_result = poker.slow.evaluateHand(hand);
    const fast_result = poker.evaluateHand(hand);

    return .{
        .slow = slow_result,
        .fast = fast_result,
        .match = slow_result == fast_result,
    };
}

// Equity benchmark for measuring Monte Carlo performance
pub fn benchmarkEquity(allocator: std.mem.Allocator, options: BenchmarkOptions) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer_wrapper = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer_wrapper.interface;
    defer stdout.flush() catch {};

    // Generate test hands for equity calculations
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    // Test different scenarios
    const scenarios = [_]struct {
        name: []const u8,
        iterations: u32,
    }{
        .{ .name = "Preflop (no board)", .iterations = 10000 },
        .{ .name = "Flop (3 cards)", .iterations = 5000 },
        .{ .name = "Turn (4 cards)", .iterations = 2000 },
        .{ .name = "River (5 cards)", .iterations = 1000 },
    };

    try stdout.print("\nEquity Calculation Performance\n", .{});
    try stdout.print("==================================================================\n", .{});
    try stdout.print("Scenario           | Simulations | Time (ms) | Sims/sec (millions)\n", .{});
    try stdout.print("-------------------|-------------|-----------|--------------------\n", .{});

    for (scenarios) |scenario| {
        // Generate random hole cards for both players
        var hero_cards: u64 = 0;
        var villain_cards: u64 = 0;
        var board_cards: [5]u64 = undefined;
        var board_size: usize = 0;

        // Generate non-conflicting cards
        var used_cards: u64 = 0;

        // Hero cards (2 cards)
        for (0..2) |_| {
            var new_card: u64 = undefined;
            while (true) {
                new_card = @as(u64, 1) << @intCast(rng.intRangeAtMost(u6, 0, 51));
                if ((new_card & used_cards) == 0) break;
            }
            hero_cards |= new_card;
            used_cards |= new_card;
        }

        // Villain cards (2 cards)
        for (0..2) |_| {
            var new_card: u64 = undefined;
            while (true) {
                new_card = @as(u64, 1) << @intCast(rng.intRangeAtMost(u6, 0, 51));
                if ((new_card & used_cards) == 0) break;
            }
            villain_cards |= new_card;
            used_cards |= new_card;
        }

        // Board cards based on scenario
        if (std.mem.eql(u8, scenario.name, "Flop (3 cards)")) {
            board_size = 3;
        } else if (std.mem.eql(u8, scenario.name, "Turn (4 cards)")) {
            board_size = 4;
        } else if (std.mem.eql(u8, scenario.name, "River (5 cards)")) {
            board_size = 5;
        }

        for (0..board_size) |i| {
            var new_card: u64 = undefined;
            while (true) {
                new_card = @as(u64, 1) << @intCast(rng.intRangeAtMost(u6, 0, 51));
                if ((new_card & used_cards) == 0) break;
            }
            board_cards[i] = new_card;
            used_cards |= new_card;
        }

        // Run benchmark
        const start = std.time.nanoTimestamp();
        const result = try poker.monteCarlo(hero_cards, villain_cards, board_cards[0..board_size], scenario.iterations, rng, allocator);
        const end = std.time.nanoTimestamp();

        const elapsed_ns = @as(f64, @floatFromInt(end - start));
        const elapsed_ms = elapsed_ns / 1_000_000.0;
        const sims_per_sec = @as(f64, @floatFromInt(scenario.iterations)) / (elapsed_ns / 1_000_000_000.0);

        try stdout.print("{s:<18} | {d:>11} | {d:>9.2} | {d:>18.2}\n", .{
            scenario.name,
            scenario.iterations,
            elapsed_ms,
            sims_per_sec / 1_000_000.0,
        });

        if (options.verbose) {
            try stdout.print("  Hero equity: {d:.2}%\n", .{result.equity() * 100.0});
        }
    }

    // Multi-way equity benchmark
    try stdout.print("\nMulti-way Equity (3+ players)\n", .{});
    try stdout.print("----------------------------------------------------------------\n", .{});

    const player_counts = [_]usize{ 3, 4, 6, 9 };
    const multiway_iterations = 5000;

    for (player_counts) |num_players| {
        // Generate hands for all players
        const hands = try allocator.alloc(u64, num_players);
        defer allocator.free(hands);

        var used: u64 = 0;
        for (hands) |*hand| {
            hand.* = 0; // Initialize to empty
            // Generate 2 cards for this player
            for (0..2) |_| {
                var new_card: u64 = undefined;
                while (true) {
                    new_card = @as(u64, 1) << @intCast(rng.intRangeAtMost(u6, 0, 51));
                    if ((new_card & used) == 0) break;
                }
                hand.* |= new_card;
                used |= new_card;
            }
        }

        const start = std.time.nanoTimestamp();

        _ = try poker.multiway(hands, &.{}, // No board
            multiway_iterations, rng, allocator);

        const end = std.time.nanoTimestamp();
        const elapsed_ns = @as(f64, @floatFromInt(end - start));
        const elapsed_ms = elapsed_ns / 1_000_000.0;
        const sims_per_sec = @as(f64, @floatFromInt(multiway_iterations)) / (elapsed_ns / 1_000_000_000.0);

        try stdout.print("{d} players         | {d:>11} | {d:>9.2} | {d:>18.2}\n", .{
            num_players,
            multiway_iterations,
            elapsed_ms,
            sims_per_sec / 1_000_000.0,
        });
    }
}

// Benchmark different batch sizes
pub fn benchmarkBatchSizes(allocator: std.mem.Allocator) !void {
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer_wrapper = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer_wrapper.interface;
    defer stdout.flush() catch {};

    // Generate test hands
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    const num_test_hands = 1_600_000;
    const test_hands = try allocator.alloc(u64, num_test_hands);
    defer allocator.free(test_hands);

    for (test_hands) |*hand| {
        hand.* = poker.generateRandomHand(&rng);
    }

    // Warmup caches
    try stdout.print("Warming up caches...\n", .{});
    warmupCaches(test_hands);

    try stdout.print("\nBatch Size Performance Comparison\n", .{});
    try stdout.print("==================================================================\n", .{});
    try stdout.print("Batch Size | ns/hand | Million hands/sec | Speedup vs single\n", .{});
    try stdout.print("-----------|---------|-------------------|------------------\n", .{});

    // Benchmark single hand for baseline
    const single_time = benchmarkSingleHand(test_hands, 1000000);
    try stdout.print("{d:>10} | {d:>7.2} | {d:>17.1} | {d:>16.2}x\n", .{ 1, single_time, 1000.0 / single_time, 1.0 });

    // Test different batch sizes
    const batch_sizes = [_]usize{ 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64 };

    inline for (batch_sizes) |batch_size| {
        const iterations = @max(100000, 1000000 / batch_size);
        const time_per_hand = try benchmarkBatchSizeGeneric(batch_size, test_hands, iterations);
        const speedup = single_time / time_per_hand;

        try stdout.print("{d:>10} | {d:>7.2} | {d:>17.1} | {d:>16.2}x\n", .{ batch_size, time_per_hand, 1000.0 / time_per_hand, speedup });
    }
}

fn benchmarkBatchSizeGeneric(comptime batchSize: usize, test_hands: []const u64, iterations: u32) !f64 {
    var timer = try std.time.Timer.start();

    // Create multiple batches from test hands to avoid cache artifacts
    var checksum: u64 = 0;

    // Run multiple times for more accurate measurement
    var best_time: f64 = std.math.inf(f64);

    for (0..3) |_| {
        const start = timer.read();

        for (0..iterations) |iter| {
            // Use different hands for each iteration
            const offset = (iter * batchSize) % (test_hands.len - batchSize);

            var batch_array: [batchSize]u64 = undefined;
            for (0..batchSize) |i| {
                batch_array[i] = test_hands[offset + i];
            }
            const batch: @Vector(batchSize, u64) = batch_array;

            const results = poker.evaluateBatch(batchSize, batch);

            // Prevent optimization
            inline for (0..batchSize) |i| {
                checksum +%= results[i];
            }
        }

        const elapsed = timer.read() - start;
        const total_hands = iterations * batchSize;
        const ns_per_hand = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(total_hands));

        if (ns_per_hand < best_time) {
            best_time = ns_per_hand;
        }
    }

    std.mem.doNotOptimizeAway(checksum);
    return best_time;
}
