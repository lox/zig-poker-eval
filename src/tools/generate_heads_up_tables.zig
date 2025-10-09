// Optimized heads-up equity table generator
// Uses BoardContext + SIMD batch evaluation + multi-threading
// Runtime: ~15 minutes for exact enumeration of all 354 billion showdowns

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
    var villain_holes_buffer: [max_villain_hands]poker.Hand = undefined;
    var hero_holes_buffer: [max_villain_hands]poker.Hand = undefined;
    var results_buffer: [max_villain_hands]i8 = undefined;

    // Generate all villain hands once
    var villain_count: usize = 0;
    for (0..available_count - 1) |v1| {
        for (v1 + 1..available_count) |v2| {
            const villain_card1 = @as(u64, 1) << @intCast(available[v1]);
            const villain_card2 = @as(u64, 1) << @intCast(available[v2]);
            villain_holes_buffer[villain_count] = villain_card1 | villain_card2;
            hero_holes_buffer[villain_count] = hero_hole;
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

                        // Create board context once for this board
                        const board_with_hero = hero_hole | board;
                        const ctx = poker.initBoardContext(board_with_hero);

                        // Filter villain hands that don't conflict with board
                        var valid_villain_count: usize = 0;
                        for (0..villain_count) |v| {
                            const villain_hole = villain_holes_buffer[v];
                            if ((villain_hole & board) == 0) {
                                villain_holes_buffer[valid_villain_count] = villain_hole;
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
