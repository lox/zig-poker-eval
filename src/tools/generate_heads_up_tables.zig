// Tool to generate heads-up equity tables
// Calculates EXACT equity for all 169 starting hands vs random opponent
// Output: src/heads_up_tables.zig with hardcoded values
//
// Uses full enumeration for perfect accuracy:
// - Total: ~354 billion hand evaluations
// - Runtime: ~10 hours at 10M evals/sec
// - Accuracy: 100% exact (no sampling error)

const std = @import("std");
const poker = @import("poker");
const print = std.debug.print;

// The 169 unique starting hands
const StartingHand = struct {
    rank1: u8,
    rank2: u8,
    suited: bool,
    index: u8,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("\n========================================\n", .{});
    print("Heads-Up Equity Table Generator\n", .{});
    print("========================================\n\n", .{});

    // Generate all 169 starting hands
    const hands = generateStartingHands();

    // Estimate time
    const total_evals = estimateTotalEvaluations();
    const estimated_seconds = @as(f64, @floatFromInt(total_evals)) / 10_000_000.0; // 10M evals/sec
    print("Estimated evaluations: {}\n", .{total_evals});
    print("Estimated time at 10M evals/sec: {d:.1} seconds ({d:.1} minutes)\n\n", .{
        estimated_seconds,
        estimated_seconds / 60.0,
    });

    // Calculate equities
    print("Calculating exact equities for 169 starting hands...\n", .{});
    print("This will take approximately 10 hours. Progress will be shown.\n\n", .{});

    var equities: [169][3]u32 = undefined;

    const start_time = std.time.milliTimestamp();
    var total_evaluations: u64 = 0;

    for (hands, 0..) |hand, i| {
        const hand_start = std.time.milliTimestamp();

        print("Hand {}/{}: ", .{ i + 1, 169 });
        // Print hand name if notable
        const name = getHandComment(@intCast(i));
        if (name.len > 0) {
            print("{s} ", .{name});
        }
        print("calculating...\n", .{});

        const equity = try calculateEquityVsRandom(hand, allocator);
        equities[i] = equity;

        const hand_elapsed = std.time.milliTimestamp() - hand_start;
        const hand_total = equity.wins + equity.losses + equity.ties;
        total_evaluations += hand_total;

        print("  Completed in {d:.1}s - Win: {d:.2}% Tie: {d:.2}%\n", .{
            @as(f64, @floatFromInt(hand_elapsed)) / 1000.0,
            @as(f64, @floatFromInt(equity.wins)) * 100.0 / @as(f64, @floatFromInt(hand_total)),
            @as(f64, @floatFromInt(equity.ties)) * 100.0 / @as(f64, @floatFromInt(hand_total)),
        });

        // Estimate remaining time
        if (i > 0) {
            const elapsed_total = std.time.milliTimestamp() - start_time;
            const avg_per_hand = @as(f64, @floatFromInt(elapsed_total)) / @as(f64, @floatFromInt(i + 1));
            const remaining_hands = 169 - (i + 1);
            const estimated_remaining = avg_per_hand * @as(f64, @floatFromInt(remaining_hands)) / 1000.0 / 60.0;

            print("  Progress: {}/169 hands, ~{d:.1} minutes remaining\n\n", .{ i + 1, estimated_remaining });
        }
    }

    const elapsed_ms = std.time.milliTimestamp() - start_time;
    const hours = @as(f64, @floatFromInt(elapsed_ms)) / 1000.0 / 3600.0;
    print("\nCompleted {} billion evaluations in {d:.2} hours\n", .{ total_evaluations / 1_000_000_000, hours });

    // Convert to win percentages (x1000 for precision)
    var equity_table: [169][2]u16 = undefined;
    for (equities, 0..) |eq, i| {
        const total = eq[0] + eq[1] + eq[2];
        if (total > 0) {
            // Store as permille (parts per thousand)
            equity_table[i][0] = @intCast((eq[0] * 1000) / total); // Win rate
            equity_table[i][1] = @intCast((eq[2] * 1000) / total); // Tie rate
        } else {
            equity_table[i] = .{ 500, 0 };
        }
    }

    // Write output file
    print("\nWriting to src/heads_up_tables.zig...\n", .{});
    try writeTableFile(equity_table);

    print("✓ Table generation complete!\n", .{});
}

fn generateStartingHands() [169]StartingHand {
    var hands: [169]StartingHand = undefined;
    var index: u8 = 0;

    // Generate in the same order as HandIndex
    for (0..13) |r1| {
        for (0..13) |r2| {
            if (r1 == r2) {
                // Pocket pair
                hands[index] = .{
                    .rank1 = @intCast(r1),
                    .rank2 = @intCast(r2),
                    .suited = false,
                    .index = index,
                };
                index += 1;
            } else if (r1 > r2) {
                // Suited (upper triangle)
                hands[index] = .{
                    .rank1 = @intCast(r1),
                    .rank2 = @intCast(r2),
                    .suited = true,
                    .index = index,
                };
                index += 1;
            } else {
                // Offsuit (lower triangle)
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

fn calculateEquityVsRandom(hand: StartingHand, allocator: std.mem.Allocator) !struct { wins: u32, losses: u32, ties: u32 } {
    var wins: u64 = 0;
    var losses: u64 = 0;
    var ties: u64 = 0;

    // Create our hero hand
    const hero_hand = createHand(hand);

    // Build available cards list (all cards except hero's)
    var available = std.ArrayList(u8).init(allocator);
    defer available.deinit();
    for (0..52) |card| {
        if ((hero_hand & (@as(u64, 1) << @intCast(card))) == 0) {
            try available.append(@intCast(card));
        }
    }

    const n_available = available.items.len; // Should be 50
    var evaluations: u64 = 0;

    // Enumerate all opponent hands: C(50,2) = 1,225
    for (0..n_available - 1) |v1| {
        const villain_card1 = @as(u64, 1) << @intCast(available.items[v1]);

        for (v1 + 1..n_available) |v2| {
            const villain_card2 = @as(u64, 1) << @intCast(available.items[v2]);
            const villain_hand = villain_card1 | villain_card2;

            // For each villain hand, enumerate all boards: C(48,5) = 1,712,304
            // We need to skip the 2 villain cards from the 50 available
            for (0..n_available - 4) |b1| {
                if (b1 == v1 or b1 == v2) continue;
                const board1 = @as(u64, 1) << @intCast(available.items[b1]);

                for (b1 + 1..n_available - 3) |b2| {
                    if (b2 == v1 or b2 == v2) continue;
                    const board2 = @as(u64, 1) << @intCast(available.items[b2]);

                    for (b2 + 1..n_available - 2) |b3| {
                        if (b3 == v1 or b3 == v2) continue;
                        const board3 = @as(u64, 1) << @intCast(available.items[b3]);

                        for (b3 + 1..n_available - 1) |b4| {
                            if (b4 == v1 or b4 == v2) continue;
                            const board4 = @as(u64, 1) << @intCast(available.items[b4]);

                            for (b4 + 1..n_available) |b5| {
                                if (b5 == v1 or b5 == v2) continue;
                                const board5 = @as(u64, 1) << @intCast(available.items[b5]);

                                const board = board1 | board2 | board3 | board4 | board5;

                                // Evaluate both hands
                                const hero_rank = poker.evaluateHand(hero_hand | board);
                                const villain_rank = poker.evaluateHand(villain_hand | board);

                                if (hero_rank < villain_rank) {
                                    wins += 1;
                                } else if (hero_rank > villain_rank) {
                                    losses += 1;
                                } else {
                                    ties += 1;
                                }

                                evaluations += 1;

                                // Progress tracking every 100M evaluations
                                if (evaluations % 100_000_000 == 0) {
                                    const total = wins + losses + ties;
                                    print("  Hand {}: {}M evals, win={d:.1}%\n", .{ hand.index, evaluations / 1_000_000, @as(f64, @floatFromInt(wins)) * 100.0 / @as(f64, @floatFromInt(total)) });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Return as u32 for compatibility
    return .{ .wins = @intCast(wins), .losses = @intCast(losses), .ties = @intCast(ties) };
}

fn createHand(hand: StartingHand) u64 {
    // Create a hand from the starting hand description
    if (hand.rank1 == hand.rank2) {
        // Pocket pair - use first two suits
        const card1 = hand.rank1 + (0 * 13); // clubs
        const card2 = hand.rank1 + (1 * 13); // diamonds
        return (@as(u64, 1) << card1) | (@as(u64, 1) << card2);
    } else if (hand.suited) {
        // Suited - use same suit (clubs)
        const card1 = hand.rank1 + (0 * 13);
        const card2 = hand.rank2 + (0 * 13);
        return (@as(u64, 1) << card1) | (@as(u64, 1) << card2);
    } else {
        // Offsuit - use different suits
        const card1 = hand.rank1 + (0 * 13); // clubs
        const card2 = hand.rank2 + (1 * 13); // diamonds
        return (@as(u64, 1) << card1) | (@as(u64, 1) << card2);
    }
}

fn getAvailableCard(index: usize, used_cards: u64) ?u64 {
    var count: usize = 0;
    for (0..52) |card| {
        const card_mask = @as(u64, 1) << @intCast(card);
        if ((used_cards & card_mask) == 0) {
            if (count == index) {
                return card_mask;
            }
            count += 1;
        }
    }
    return null;
}

fn estimateTotalEvaluations() u64 {
    // Full enumeration: 169 hands × C(50,2) opponents × C(48,5) boards
    // = 169 × 1,225 × 1,712,304 = 354,643,972,800
    return 169 * 1225 * 1712304;
}

fn writeTableFile(equity_table: [169][2]u16) !void {
    const file = try std.fs.cwd().createFile("src/heads_up_tables.zig", .{});
    defer file.close();
    const writer = file.writer();

    try writer.print("// Generated heads-up equity tables\n", .{});
    try writer.print("// Each entry is (win_rate_x1000, tie_rate_x1000)\n", .{});
    try writer.print("// For example, (850, 23) means 85.0% win, 2.3% tie\n", .{});
    try writer.print("//\n", .{});
    try writer.print("// Generated by: zig run src/tools/generate_heads_up_tables.zig\n", .{});
    try writer.print("// Timestamp: {}\n\n", .{std.time.milliTimestamp()});

    try writer.print("pub const PREFLOP_VS_RANDOM: [169][2]u16 = .{{\n", .{});

    for (equity_table, 0..) |eq, i| {
        // Add comment for notable hands
        const comment = getHandComment(i);
        if (comment.len > 0) {
            try writer.print("    .{{ {}, {} }}, // {s}\n", .{ eq[0], eq[1], comment });
        } else {
            try writer.print("    .{{ {}, {} }},\n", .{ eq[0], eq[1] });
        }
    }

    try writer.print("}};\n", .{});
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
