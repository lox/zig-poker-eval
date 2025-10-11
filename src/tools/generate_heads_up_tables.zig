// Heads-Up Equity Table Generator
// ================================
//
// PURPOSE:
// Generates precomputed equity tables for all 169 unique Texas Hold'em starting hands
// when playing heads-up (1v1) against a random opponent. These tables enable instant
// equity lookups during gameplay without requiring expensive real-time calculations.
//
// WHY WE NEED THIS:
// Calculating exact equity requires enumerating millions of possible outcomes per hand.
// For real-time applications (poker bots, training tools, analysis software), doing this
// calculation on-demand is too slow. By precomputing these values once and storing them
// in a lookup table, we can get instant O(1) equity lookups.
//
// ALGORITHM:
// For each of the 169 starting hands, we perform EXACT enumeration (not Monte Carlo):
//
// 1. Fix hero's 2 hole cards (e.g., Ac Ad for pocket aces)
// 2. Enumerate all possible villain hands: C(50,2) = 1,225 combinations
// 3. For each villain hand, enumerate all boards: C(48,5) = 1,712,304 combinations
// 4. For each scenario, determine winner using our fast evaluator
// 5. Sum up wins/losses/ties to calculate exact equity
//
// Total scenarios per hand: 1,225 × 1,712,304 = 2,097,572,400
// Total for all 169 hands: ~354 billion evaluations
//
// OPTIMIZATIONS:
// - BoardContext: Precompute board analysis once, reuse for all villain hands
// - SIMD Batch Evaluation: Process multiple hands in parallel using CPU vectorization
// - Multi-threading: Distribute the 169 hands across available CPU cores
// - Smart buffering: Reuse allocated arrays to minimize memory allocation
//
// KEY FIXES FROM INITIAL VERSION:
// 1. BoardContext must contain ONLY the 5 board cards, not hero's hole cards
// 2. Villain hands must be copied fresh from master list for each board (not overwritten)
//
// OUTPUT:
// Generates src/heads_up_tables.zig with a 169-entry lookup table where each entry
// contains (win_rate × 1000, tie_rate × 1000) as u16 values for space efficiency.
// Example: {852, 18} means 85.2% win, 1.8% tie, 13.0% loss (implied)
//
// VALIDATION:
// Before writing output, validates against known poker equities to ensure correctness.
// If any validation fails, the generation aborts to prevent bad data from being used.
//
// Runtime: ~15-25 minutes on modern multi-core CPU

const std = @import("std");
const poker = @import("poker");
const print = std.debug.print;

const HandResult = struct {
    hand: poker.StartingHand,
    wins: u64,
    losses: u64,
    ties: u64,
};

const ThreadContext = struct {
    hands: []const poker.StartingHand,
    results: []HandResult,
    start_idx: usize,
    end_idx: usize,
    thread_id: u32,
    wait_group: *std.Thread.WaitGroup,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n========================================\n", .{});
    print("Heads-Up Equity Table Generator\n", .{});
    print("========================================\n\n", .{});

    const hands = poker.ALL_STARTING_HANDS;

    // Estimate performance
    const boards_per_hand = 1712304; // C(48,5)
    const villain_hands_per_board = 1225; // C(50,2)
    const total_showdowns = 169 * boards_per_hand * villain_hands_per_board;
    print("Total showdowns: {} billion\n", .{total_showdowns / 1_000_000_000});

    const start_time = std.time.milliTimestamp();

    // Get thread count
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    print("Using {} threads\n\n", .{thread_count});

    // Allocate results
    const results = try allocator.alloc(HandResult, 169);
    defer allocator.free(results);

    // Initialize results
    for (results, 0..) |*result, i| {
        result.* = .{
            .hand = hands[i],
            .wins = 0,
            .losses = 0,
            .ties = 0,
        };
    }

    // Divide work among threads
    const hands_per_thread: usize = @as(usize, 169) / thread_count;
    const remaining_hands: usize = @as(usize, 169) % thread_count;

    var contexts = try allocator.alloc(ThreadContext, thread_count);
    defer allocator.free(contexts);

    var wait_group = std.Thread.WaitGroup{};

    for (0..thread_count) |i| {
        const start_idx = i * hands_per_thread + @min(i, remaining_hands);
        const extra = if (i < remaining_hands) @as(usize, 1) else 0;
        const end_idx = start_idx + hands_per_thread + extra;

        contexts[i] = ThreadContext{
            .hands = &hands,
            .results = results,
            .start_idx = start_idx,
            .end_idx = end_idx,
            .thread_id = @intCast(i),
            .wait_group = &wait_group,
        };

        wait_group.start();
        _ = try std.Thread.spawn(.{}, workerThread, .{&contexts[i]});
    }

    wait_group.wait();

    const elapsed_ms = std.time.milliTimestamp() - start_time;
    const minutes = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0 / 60.0;
    print("\n✓ Completed in {d:.1} minutes\n", .{minutes});

    // Convert to win percentages (using u64 intermediate to avoid overflow)
    var equity_table: [169][2]u16 = undefined;
    for (results, 0..) |result, i| {
        const total = result.wins + result.losses + result.ties;
        if (total > 0) {
            // Use explicit u64 cast to prevent overflow when multiplying by 1000
            // (result.wins can be ~2 billion from exactVsRandom)
            const wins_u64: u64 = result.wins;
            const ties_u64: u64 = result.ties;
            equity_table[i][0] = @intCast((wins_u64 * 1000) / total);
            equity_table[i][1] = @intCast((ties_u64 * 1000) / total);
        } else {
            equity_table[i] = .{ 500, 0 };
        }
    }

    // Validate against known values before writing
    print("\nValidating against known equities...\n", .{});
    const valid = try validateKnownEquities(equity_table);

    if (!valid) {
        print("\n❌ VALIDATION FAILED: Generated values don't match known equities!\n", .{});
        print("   Please check the calculation logic.\n", .{});
        return error.ValidationFailed;
    }

    print("✓ All validations passed!\n\n", .{});

    // Write output
    print("Writing to src/heads_up_tables.zig...\n", .{});
    try writeTableFile(equity_table);
    print("✓ Complete!\n", .{});
}

fn workerThread(ctx: *ThreadContext) void {
    defer ctx.wait_group.finish();

    // Thread-local allocator for equity calculations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    for (ctx.start_idx..ctx.end_idx) |i| {
        const hand = ctx.hands[i];
        const hero_hole = createHand(hand);

        // Use the new exactVsRandom function from equity module
        const equity_result = poker.equity.exactVsRandom(hero_hole, allocator) catch |err| {
            print("Thread {} error on hand {}: {}\n", .{ ctx.thread_id, i, err });
            return;
        };

        ctx.results[i] = HandResult{
            .hand = hand,
            .wins = @intCast(equity_result.wins),
            .losses = equity_result.total_simulations - equity_result.wins - equity_result.ties,
            .ties = @intCast(equity_result.ties),
        };

        // Progress reporting
        print("Thread {} - Hand {}/169: Win={d:.2}% Tie={d:.2}%\n", .{
            ctx.thread_id,
            i + 1,
            equity_result.winRate() * 100.0,
            equity_result.tieRate() * 100.0,
        });
    }
}

// Old calculateEquityVsRandomFast function removed - now using equity.exactVsRandom()

fn createHand(hand: poker.StartingHand) u64 {
    const high_val = @intFromEnum(hand.high);
    const low_val = @intFromEnum(hand.low);

    if (high_val == low_val) {
        // Pocket pair - use clubs and diamonds
        const card1 = high_val + (0 * 13); // clubs
        const card2 = high_val + (1 * 13); // diamonds
        return (@as(u64, 1) << @intCast(card1)) | (@as(u64, 1) << @intCast(card2));
    } else if (hand.suited) {
        // Suited - both cards same suit (clubs)
        const card1 = high_val + (0 * 13);
        const card2 = low_val + (0 * 13);
        return (@as(u64, 1) << @intCast(card1)) | (@as(u64, 1) << @intCast(card2));
    } else {
        // Offsuit - different suits (clubs and diamonds)
        const card1 = high_val + (0 * 13);
        const card2 = low_val + (1 * 13);
        return (@as(u64, 1) << @intCast(card1)) | (@as(u64, 1) << @intCast(card2));
    }
}

fn writeTableFile(equity_table: [169][2]u16) !void {
    const file = try std.fs.cwd().createFile("src/heads_up_tables.zig", .{});
    defer file.close();

    var buffer: [32768]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);
    const writer = fbs.writer();

    try writer.print("// Generated heads-up equity tables\n", .{});
    try writer.print("// Each entry is (win_rate_x1000, tie_rate_x1000)\n", .{});
    try writer.print("// For example, (850, 23) means 85.0% win, 2.3% tie\n", .{});
    try writer.print("//\n", .{});
    try writer.print("// Generated by: zig run src/tools/generate_heads_up_tables_fast.zig\n", .{});
    try writer.print("// Timestamp: {}\n\n", .{std.time.milliTimestamp()});

    try writer.print("pub const PREFLOP_VS_RANDOM: [169][2]u16 = .{{\n", .{});

    for (equity_table, 0..) |eq, i| {
        const comment = getHandComment(@intCast(i));
        if (comment.len > 0) {
            try writer.print("    .{{ {}, {} }}, // {s}\n", .{ eq[0], eq[1], comment });
        } else {
            try writer.print("    .{{ {}, {} }},\n", .{ eq[0], eq[1] });
        }
    }

    try writer.print("}};\n", .{});

    try file.writeAll(fbs.getWritten());
}

fn getHandComment(index: u8) []const u8 {
    const hand = poker.StartingHand.fromIndex(index);
    const notation = hand.toNotation();

    // Only show comments for pocket pairs and notable hands
    if (hand.high == hand.low) {
        return notation; // All pocket pairs
    } else if (index == 167 or index == 155) {
        return notation; // AKs and AKo
    } else {
        return "";
    }
}

fn validateKnownEquities(equity_table: [169][2]u16) !bool {
    const KnownEquity = struct {
        index: u8,
        hand: []const u8,
        expected_win: u16, // x10 (852 = 85.2%)
        tolerance: u16, // x10 (10 = 1.0%)
    };

    // Known equities from poker literature and calculators
    const known_values = [_]KnownEquity{
        .{ .index = 168, .hand = "AA", .expected_win = 852, .tolerance = 10 },
        .{ .index = 154, .hand = "KK", .expected_win = 824, .tolerance = 10 },
        .{ .index = 140, .hand = "QQ", .expected_win = 799, .tolerance = 10 },
        .{ .index = 126, .hand = "JJ", .expected_win = 775, .tolerance = 10 },
        .{ .index = 112, .hand = "TT", .expected_win = 750, .tolerance = 10 },
        .{ .index = 98, .hand = "99", .expected_win = 721, .tolerance = 10 },
        .{ .index = 84, .hand = "88", .expected_win = 691, .tolerance = 10 },
        .{ .index = 70, .hand = "77", .expected_win = 662, .tolerance = 10 },
        .{ .index = 56, .hand = "66", .expected_win = 633, .tolerance = 10 },
        .{ .index = 42, .hand = "55", .expected_win = 603, .tolerance = 10 },
        .{ .index = 28, .hand = "44", .expected_win = 570, .tolerance = 10 },
        .{ .index = 14, .hand = "33", .expected_win = 537, .tolerance = 10 },
        .{ .index = 0, .hand = "22", .expected_win = 503, .tolerance = 10 },
        .{ .index = 167, .hand = "AKs", .expected_win = 670, .tolerance = 10 },
        .{ .index = 155, .hand = "AKo", .expected_win = 653, .tolerance = 10 },
    };

    var all_valid = true;
    print("\n  Hand  | Expected | Generated | Diff  | Status\n", .{});
    print("  ------|----------|-----------|-------|--------\n", .{});

    for (known_values) |kv| {
        const generated_win = equity_table[kv.index][0];
        const diff = if (generated_win > kv.expected_win)
            generated_win - kv.expected_win
        else
            kv.expected_win - generated_win;

        const is_valid = diff <= kv.tolerance;
        const status = if (is_valid) "✓ PASS" else "✗ FAIL";

        print("  {s:5} | {d:7.1}% | {d:8.1}% | {d:4.1}% | {s}\n", .{
            kv.hand,
            @as(f64, @floatFromInt(kv.expected_win)) / 10.0,
            @as(f64, @floatFromInt(generated_win)) / 10.0,
            @as(f64, @floatFromInt(diff)) / 10.0,
            status,
        });

        if (!is_valid) {
            all_valid = false;
            print("    ⚠️  {s} equity {d:.1}% is outside tolerance of {d:.1}%±{d:.1}%\n", .{
                kv.hand,
                @as(f64, @floatFromInt(generated_win)) / 10.0,
                @as(f64, @floatFromInt(kv.expected_win)) / 10.0,
                @as(f64, @floatFromInt(kv.tolerance)) / 10.0,
            });
        }
    }

    return all_valid;
}
