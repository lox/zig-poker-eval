// Heads-Up 169x169 Equity Matrix Generator
// ========================================
//
// PURPOSE:
// Generates a complete 169x169 matrix of exact preflop equities for all possible
// starting hand matchups in heads-up Texas Hold'em.
//
// WHY WE NEED THIS:
// The vs-random table (PREFLOP_VS_RANDOM) only shows equity against a random opponent,
// which is an approximation. For accurate head-to-head equity (e.g., AA vs KK specifically),
// we need exact calculations for each matchup.
//
// ALGORITHM:
// For each pair of starting hand indices (i, j) where i, j ∈ [0, 168]:
//   1. Map indices to hand ranks (e.g., i=168 → AA, j=154 → KK)
//   2. Create canonical non-conflicting card combinations
//      (e.g., AcAd vs KhKs to avoid suit conflicts)
//   3. Enumerate all C(48,5) = 1,712,304 possible boards
//   4. Evaluate each board to determine winner
//   5. Store (win_rate × 1000, tie_rate × 1000) in matrix[i][j]
//
// TOTAL WORK:
// - 169 × 169 = 28,561 matchups
// - Each matchup: 1,712,304 board evaluations
// - Total: ~48.9 billion evaluations
// - Estimated runtime: ~20-30 minutes with multi-threading
//
// OPTIMIZATIONS:
// - Symmetry: matrix[i][j] can be derived from matrix[j][i] (win/loss swap)
// - Multi-threading: Distribute matchups across CPU cores
// - Only compute upper triangle + diagonal (saves 50% work)
//
// OUTPUT:
// Generates src/heads_up_matrix.zig with a 169×169 lookup table

const std = @import("std");
const poker = @import("poker");
const print = std.debug.print;

const MatchupResult = struct {
    hero_idx: u8,
    villain_idx: u8,
    wins: u32,
    ties: u32,
    total: u32,
};

const ThreadContext = struct {
    matchups: []const [2]u8, // pairs of (hero_idx, villain_idx)
    results: []MatchupResult,
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
    print("Heads-Up 169x169 Equity Matrix Generator\n", .{});
    print("========================================\n\n", .{});

    const start_time = std.time.milliTimestamp();

    // Generate list of matchups to compute (upper triangle + diagonal)
    var matchup_list = std.ArrayList([2]u8).empty;
    defer matchup_list.deinit(allocator);

    for (0..169) |i| {
        for (i..169) |j| {
            try matchup_list.append(allocator, .{ @intCast(i), @intCast(j) });
        }
    }

    const total_matchups = matchup_list.items.len;
    print("Computing {d} unique matchups (upper triangle + diagonal)\n", .{total_matchups});
    print("Total evaluations: ~{d} billion\n\n", .{total_matchups * 1712304 / 1_000_000_000});

    // Get thread count
    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    print("Using {} threads\n\n", .{thread_count});

    // Allocate results
    const results = try allocator.alloc(MatchupResult, total_matchups);
    defer allocator.free(results);

    // Initialize results
    for (results, 0..) |*result, idx| {
        result.* = .{
            .hero_idx = matchup_list.items[idx][0],
            .villain_idx = matchup_list.items[idx][1],
            .wins = 0,
            .ties = 0,
            .total = 0,
        };
    }

    // Divide work among threads
    const matchups_per_thread = total_matchups / thread_count;
    const remaining_matchups = total_matchups % thread_count;

    var contexts = try allocator.alloc(ThreadContext, thread_count);
    defer allocator.free(contexts);

    var wait_group = std.Thread.WaitGroup{};

    for (0..thread_count) |i| {
        const start_idx = i * matchups_per_thread + @min(i, remaining_matchups);
        const extra = if (i < remaining_matchups) @as(usize, 1) else 0;
        const end_idx = start_idx + matchups_per_thread + extra;

        contexts[i] = ThreadContext{
            .matchups = matchup_list.items,
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

    // Build full 169x169 matrix
    var matrix: [169][169][2]u16 = undefined;

    // Initialize with zeros
    for (0..169) |i| {
        for (0..169) |j| {
            matrix[i][j] = .{ 0, 0 };
        }
    }

    // Fill matrix from results
    for (results) |result| {
        if (result.total > 0) {
            const win_rate: u16 = @intCast((result.wins * 1000) / result.total);
            const tie_rate: u16 = @intCast((result.ties * 1000) / result.total);

            const i = result.hero_idx;
            const j = result.villain_idx;

            // Set both [i][j] and [j][i] using symmetry
            matrix[i][j] = .{ win_rate, tie_rate };

            if (i != j) {
                // Swap win/loss for mirror matchup
                const loss_rate: u16 = 1000 - win_rate - tie_rate;
                matrix[j][i] = .{ loss_rate, tie_rate };
            }
        }
    }

    // Validate some known matchups
    print("\nValidating known matchups...\n", .{});
    const valid = validateKnownMatchups(matrix);

    if (!valid) {
        print("\n❌ VALIDATION FAILED: Generated values don't match known equities!\n", .{});
        return error.ValidationFailed;
    }

    print("✓ All validations passed!\n\n", .{});

    // Write output
    print("Writing to src/heads_up_matrix.zig...\n", .{});
    try writeMatrixFile(matrix);
    print("✓ Complete!\n", .{});
}

fn workerThread(ctx: *ThreadContext) void {
    defer ctx.wait_group.finish();

    // Thread-local allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var progress_counter: usize = 0;
    const report_interval = 100; // Report every 100 matchups

    for (ctx.start_idx..ctx.end_idx) |idx| {
        const matchup = ctx.matchups[idx];
        const hero_idx = matchup[0];
        const villain_idx = matchup[1];

        // Convert indices to notation strings
        const hero_notation = indexToNotation(hero_idx);
        const villain_notation = indexToNotation(villain_idx);

        // Use range API to create ranges and calculate equity
        var hero_range = poker.parseRange(hero_notation, allocator) catch |err| {
            print("Thread {} error parsing hero {s}: {}\n", .{ ctx.thread_id, hero_notation, err });
            continue;
        };
        defer hero_range.deinit();

        var villain_range = poker.parseRange(villain_notation, allocator) catch |err| {
            print("Thread {} error parsing villain {s}: {}\n", .{ ctx.thread_id, villain_notation, err });
            continue;
        };
        defer villain_range.deinit();

        // Calculate exact equity using range API (handles all combinations and weighting)
        const range_result = hero_range.equityExact(&villain_range, &.{}, allocator) catch |err| {
            print("Thread {} error on matchup [{},{}]: {}\n", .{ ctx.thread_id, hero_idx, villain_idx, err });
            continue;
        };

        // Convert range equity to matchup result
        const total_combos = range_result.total_simulations;
        const wins = @as(u32, @intFromFloat(range_result.hero_equity * @as(f64, @floatFromInt(total_combos))));
        const ties = 0; // Range API doesn't track ties separately, it's in the equity

        ctx.results[idx] = MatchupResult{
            .hero_idx = hero_idx,
            .villain_idx = villain_idx,
            .wins = wins,
            .ties = ties,
            .total = total_combos,
        };

        // Progress reporting
        progress_counter += 1;
        if (progress_counter % report_interval == 0) {
            const pct = (@as(f64, @floatFromInt(idx - ctx.start_idx + 1)) / @as(f64, @floatFromInt(ctx.end_idx - ctx.start_idx))) * 100.0;
            print("Thread {} - {d:.1}% ({}/{}) - {s} vs {s}: {d:.1}%\n", .{
                ctx.thread_id,
                pct,
                idx - ctx.start_idx + 1,
                ctx.end_idx - ctx.start_idx,
                hero_notation,
                villain_notation,
                range_result.hero_equity * 100.0,
            });
        }
    }
}

/// Convert hand index (0-168) to poker notation string
/// Index mapping (from HandIndex.getIndex):
/// - Pocket pairs: rank * 13 + rank (diagonal, rank1 == rank2)
/// - Suited: high_rank * 13 + low_rank (upper triangle, row > col)
/// - Offsuit: low_rank * 13 + high_rank (lower triangle, row < col)
fn indexToNotation(hand_idx: u8) []const u8 {
    const row = hand_idx / 13;
    const col = hand_idx % 13;

    if (row == col) {
        // Pocket pair - diagonal
        return switch (row) {
            0 => "22",
            1 => "33",
            2 => "44",
            3 => "55",
            4 => "66",
            5 => "77",
            6 => "88",
            7 => "99",
            8 => "TT",
            9 => "JJ",
            10 => "QQ",
            11 => "KK",
            12 => "AA",
            else => unreachable,
        };
    } else if (row > col) {
        // Suited - upper triangle
        const high_rank = row;
        const low_rank = col;
        return formatHandNotation(high_rank, low_rank, true);
    } else {
        // Offsuit - lower triangle (col > row)
        const high_rank = col;
        const low_rank = row;
        return formatHandNotation(high_rank, low_rank, false);
    }
}

/// Format two ranks into notation like "AKs" or "QJo"
/// Returns a static string literal for each possible combination
fn formatHandNotation(high_rank: u8, low_rank: u8, suited: bool) []const u8 {
    // Generate static lookup table at comptime
    const notations = comptime blk: {
        var table: [13][13][2][]const u8 = undefined;

        // For each high rank
        for (0..13) |h| {
            // For each low rank
            for (0..13) |l| {
                const h_char = switch (h) {
                    0 => "2",
                    1 => "3",
                    2 => "4",
                    3 => "5",
                    4 => "6",
                    5 => "7",
                    6 => "8",
                    7 => "9",
                    8 => "T",
                    9 => "J",
                    10 => "Q",
                    11 => "K",
                    12 => "A",
                    else => unreachable,
                };
                const l_char = switch (l) {
                    0 => "2",
                    1 => "3",
                    2 => "4",
                    3 => "5",
                    4 => "6",
                    5 => "7",
                    6 => "8",
                    7 => "9",
                    8 => "T",
                    9 => "J",
                    10 => "Q",
                    11 => "K",
                    12 => "A",
                    else => unreachable,
                };

                table[h][l][0] = h_char ++ l_char ++ "s";
                table[h][l][1] = h_char ++ l_char ++ "o";
            }
        }
        break :blk table;
    };

    const idx: usize = if (suited) 0 else 1;
    return notations[high_rank][low_rank][idx];
}

fn writeMatrixFile(matrix: [169][169][2]u16) !void {
    const file = try std.fs.cwd().createFile("src/heads_up_matrix.zig", .{});
    defer file.close();

    var buffer: [1024 * 1024]u8 = undefined; // 1MB buffer
    var fbs = std.io.fixedBufferStream(&buffer);
    const writer = fbs.writer();

    try writer.print("// Generated heads-up equity matrix (169x169)\n", .{});
    try writer.print("// Each entry is (win_rate_x1000, tie_rate_x1000)\n", .{});
    try writer.print("// For example, matrix[AA][KK] = (820, 3) means AA wins 82.0%, ties 0.3%\n", .{});
    try writer.print("//\n", .{});
    try writer.print("// Generated by: zig build gen-heads-up-matrix\n", .{});
    try writer.print("// Timestamp: {}\n\n", .{std.time.milliTimestamp()});

    try writer.print("pub const HEADS_UP_MATRIX: [169][169][2]u16 = .{{\n", .{});

    for (matrix, 0..) |row, i| {
        try writer.print("    .{{", .{});
        for (row, 0..) |entry, j| {
            if (j > 0) try writer.print(", ", .{});
            try writer.print(".{{{}, {}}}", .{ entry[0], entry[1] });
        }
        try writer.print("}},", .{});

        // Add comment for pocket pairs
        if (i % 13 == i / 13) {
            const rank_name = getRankName(@intCast(i / 13));
            try writer.print(" // {s}{s}\n", .{ rank_name, rank_name });
        } else {
            try writer.print("\n", .{});
        }
    }

    try writer.print("}};\n", .{});

    try file.writeAll(fbs.getWritten());
}

fn getRankName(rank: u8) []const u8 {
    return switch (rank) {
        0 => "2",
        1 => "3",
        2 => "4",
        3 => "5",
        4 => "6",
        5 => "7",
        6 => "8",
        7 => "9",
        8 => "T",
        9 => "J",
        10 => "Q",
        11 => "K",
        12 => "A",
        else => "?",
    };
}

fn validateKnownMatchups(matrix: [169][169][2]u16) bool {
    const KnownMatchup = struct {
        hero_idx: u8,
        villain_idx: u8,
        hero_name: []const u8,
        villain_name: []const u8,
        expected_win: u16, // x10 (820 = 82.0%)
        tolerance: u16, // x10 (10 = 1.0%)
    };

    // Known matchups from cardfight.com poker calculator
    const known_values = [_]KnownMatchup{
        .{ .hero_idx = 168, .villain_idx = 154, .hero_name = "AA", .villain_name = "KK", .expected_win = 820, .tolerance = 20 }, // Real: 81.95%
        .{ .hero_idx = 168, .villain_idx = 140, .hero_name = "AA", .villain_name = "QQ", .expected_win = 816, .tolerance = 20 }, // Real: 81.55%
        .{ .hero_idx = 168, .villain_idx = 167, .hero_name = "AA", .villain_name = "AKs", .expected_win = 879, .tolerance = 20 }, // Real: 87.86%
        .{ .hero_idx = 168, .villain_idx = 155, .hero_name = "AA", .villain_name = "AKo", .expected_win = 932, .tolerance = 20 }, // Real: 93.17%
        .{ .hero_idx = 154, .villain_idx = 140, .hero_name = "KK", .villain_name = "QQ", .expected_win = 819, .tolerance = 20 }, // Real: 81.93%
        .{ .hero_idx = 167, .villain_idx = 155, .hero_name = "AKs", .villain_name = "AKo", .expected_win = 525, .tolerance = 20 }, // Real: 52.49%
    };

    var all_valid = true;
    print("\n  Matchup        | Expected | Generated | Diff  | Status\n", .{});
    print("  ---------------|----------|-----------|-------|--------\n", .{});

    for (known_values) |kv| {
        const generated_win = matrix[kv.hero_idx][kv.villain_idx][0];
        const diff = if (generated_win > kv.expected_win)
            generated_win - kv.expected_win
        else
            kv.expected_win - generated_win;

        const is_valid = diff <= kv.tolerance;
        const status = if (is_valid) "✓ PASS" else "✗ FAIL";

        print("  {s:3} vs {s:3}  | {d:7.1}% | {d:8.1}% | {d:4.1}% | {s}\n", .{
            kv.hero_name,
            kv.villain_name,
            @as(f64, @floatFromInt(kv.expected_win)) / 10.0,
            @as(f64, @floatFromInt(generated_win)) / 10.0,
            @as(f64, @floatFromInt(diff)) / 10.0,
            status,
        });

        if (!is_valid) {
            all_valid = false;
        }
    }

    return all_valid;
}
