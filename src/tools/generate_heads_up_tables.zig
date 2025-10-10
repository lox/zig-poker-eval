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

const StartingHand = struct {
    rank1: u8,
    rank2: u8,
    suited: bool,
    index: u8,
};

const HandResult = struct {
    index: u8,
    wins: u64,
    losses: u64,
    ties: u64,
};

const ThreadContext = struct {
    hands: []const StartingHand,
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
    print("FAST Heads-Up Equity Table Generator\n", .{});
    print("Using BoardContext + Batch + Threading\n", .{});
    print("========================================\n\n", .{});

    const hands = generateStartingHands();

    // Estimate performance
    const boards_per_hand = 1712304; // C(48,5)
    const villain_hands_per_board = 1225; // C(50,2)
    const total_showdowns = 169 * boards_per_hand * villain_hands_per_board;
    print("Total showdowns: {} billion\n", .{total_showdowns / 1_000_000_000});
    print("Expected runtime: 15-25 minutes\n\n", .{});

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
            .index = @intCast(i),
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
        const start_idx = i * hands_per_thread;
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

    // Convert to win percentages
    var equity_table: [169][2]u16 = undefined;
    for (results, 0..) |result, i| {
        const total = result.wins + result.losses + result.ties;
        if (total > 0) {
            equity_table[i][0] = @intCast((result.wins * 1000) / total);
            equity_table[i][1] = @intCast((result.ties * 1000) / total);
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

    for (ctx.start_idx..ctx.end_idx) |i| {
        const hand = ctx.hands[i];
        const result = calculateEquityVsRandomFast(hand) catch |err| {
            print("Thread {} error on hand {}: {}\n", .{ ctx.thread_id, i, err });
            return;
        };

        ctx.results[i] = result;

        // Progress reporting
        const total = result.wins + result.losses + result.ties;
        print("Thread {} - Hand {}/169: Win={d:.2}% Tie={d:.2}%\n", .{
            ctx.thread_id,
            i + 1,
            @as(f64, @floatFromInt(result.wins)) * 100.0 / @as(f64, @floatFromInt(total)),
            @as(f64, @floatFromInt(result.ties)) * 100.0 / @as(f64, @floatFromInt(total)),
        });
    }
}

fn calculateEquityVsRandomFast(hand: StartingHand) !HandResult {
    var wins: u64 = 0;
    var losses: u64 = 0;
    var ties: u64 = 0;

    const hero_hole = createHand(hand);

    // Build available cards (all except hero's 2 cards)
    var available: [50]u8 = undefined;
    var available_count: u8 = 0;
    for (0..52) |card_idx| {
        const card_bit = @as(u64, 1) << @intCast(card_idx);
        if ((hero_hole & card_bit) == 0) {
            available[available_count] = @intCast(card_idx);
            available_count += 1;
        }
    }

    // Pre-allocate villain hands array for batch processing
    const max_villain_hands = 1225; // C(50,2)
    var all_villain_holes: [max_villain_hands]poker.Hand = undefined;
    var villain_holes_buffer: [max_villain_hands]poker.Hand = undefined;
    var hero_holes_buffer: [max_villain_hands]poker.Hand = undefined;
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
                        const ctx = poker.initBoardContext(board);

                        // Filter villain hands that don't conflict with board
                        // Copy valid hands from all_villain_holes to villain_holes_buffer
                        var valid_villain_count: usize = 0;
                        for (0..villain_count) |v| {
                            const villain_hole = all_villain_holes[v];
                            if ((villain_hole & board) == 0) {
                                villain_holes_buffer[valid_villain_count] = villain_hole;
                                hero_holes_buffer[valid_villain_count] = hero_hole;
                                valid_villain_count += 1;
                            }
                        }

                        // Batch evaluate all valid villain hands
                        if (valid_villain_count > 0) {
                            poker.evaluateShowdownBatch(
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

    return HandResult{
        .index = hand.index,
        .wins = wins,
        .losses = losses,
        .ties = ties,
    };
}

fn generateStartingHands() [169]StartingHand {
    var hands: [169]StartingHand = undefined;
    var index: u8 = 0;

    for (0..13) |r1| {
        for (0..13) |r2| {
            if (r1 == r2) {
                hands[index] = .{
                    .rank1 = @intCast(r1),
                    .rank2 = @intCast(r2),
                    .suited = false,
                    .index = index,
                };
                index += 1;
            } else if (r1 > r2) {
                hands[index] = .{
                    .rank1 = @intCast(r1),
                    .rank2 = @intCast(r2),
                    .suited = true,
                    .index = index,
                };
                index += 1;
            } else {
                hands[index] = .{
                    .rank1 = @intCast(r2),
                    .rank2 = @intCast(r1),
                    .suited = false,
                    .index = index,
                };
                index += 1;
            }
        }
    }

    return hands;
}

fn createHand(hand: StartingHand) u64 {
    if (hand.rank1 == hand.rank2) {
        const card1 = hand.rank1 + (0 * 13);
        const card2 = hand.rank1 + (1 * 13);
        return (@as(u64, 1) << @intCast(card1)) | (@as(u64, 1) << @intCast(card2));
    } else if (hand.suited) {
        const card1 = hand.rank1 + (0 * 13);
        const card2 = hand.rank2 + (0 * 13);
        return (@as(u64, 1) << @intCast(card1)) | (@as(u64, 1) << @intCast(card2));
    } else {
        const card1 = hand.rank1 + (0 * 13);
        const card2 = hand.rank2 + (1 * 13);
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
    return switch (index) {
        0 => "22",
        14 => "33",
        28 => "44",
        42 => "55",
        56 => "66",
        70 => "77",
        84 => "88",
        98 => "99",
        112 => "TT",
        126 => "JJ",
        140 => "QQ",
        154 => "KK",
        168 => "AA",
        167 => "AKs",
        155 => "AKo",
        else => "",
    };
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
