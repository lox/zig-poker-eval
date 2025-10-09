// Tool to generate heads-up equity tables
// Calculates equity for all 169 starting hands vs random opponent
// Output: src/heads_up_tables.zig with hardcoded values
//
// Uses Monte Carlo sampling (1M simulations per hand) for practical runtime:
// - Full enumeration: ~354 billion evaluations = ~10 hours
// - Monte Carlo 1M: 169 million evaluations = ~17 seconds
// - Accuracy: < 0.1% error vs full enumeration

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
    print("Calculating equities for 169 starting hands...\n", .{});
    var equities: [169][3]u32 = undefined;

    const start_time = std.time.milliTimestamp();

    for (hands, 0..) |hand, i| {
        if (i % 10 == 0) {
            print("Progress: {}/169 hands...\n", .{i});
        }

        const equity = try calculateEquityVsRandom(hand, allocator);
        equities[i] = equity;
    }

    const elapsed_ms = std.time.milliTimestamp() - start_time;
    print("\nCompleted in {} ms ({d:.1} seconds)\n", .{ elapsed_ms, @as(f64, @floatFromInt(elapsed_ms)) / 1000.0 });

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
    // Use Monte Carlo sampling for practical generation time
    // 1 million simulations per hand gives < 0.1% error
    const SIMULATIONS = 1_000_000;

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp() + hand.index));
    const random = prng.random();

    var wins: u32 = 0;
    var losses: u32 = 0;
    var ties: u32 = 0;

    // Create our hero hand
    const hero_hand = createHand(hand);

    // Build deck without hero cards
    var deck = std.ArrayList(u8).init(allocator);
    defer deck.deinit();

    for (0..52) |card| {
        if ((hero_hand & (@as(u64, 1) << @intCast(card))) == 0) {
            try deck.append(@intCast(card));
        }
    }

    for (0..SIMULATIONS) |_| {
        // Shuffle remaining cards for this simulation
        random.shuffle(u8, deck.items);

        // Deal villain cards (first 2) and board (next 5)
        const villain_hand = (@as(u64, 1) << @intCast(deck.items[0])) |
            (@as(u64, 1) << @intCast(deck.items[1]));

        const board = (@as(u64, 1) << @intCast(deck.items[2])) |
            (@as(u64, 1) << @intCast(deck.items[3])) |
            (@as(u64, 1) << @intCast(deck.items[4])) |
            (@as(u64, 1) << @intCast(deck.items[5])) |
            (@as(u64, 1) << @intCast(deck.items[6]));

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
    }

    return .{ .wins = wins, .losses = losses, .ties = ties };
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
    // Using Monte Carlo: 169 hands × 1 million simulations
    const SIMULATIONS_PER_HAND = 1_000_000;
    return 169 * SIMULATIONS_PER_HAND;
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
