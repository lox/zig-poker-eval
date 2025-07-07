const std = @import("std");
const poker = @import("poker");

// Structure to store hand and its evaluation
const HandResult = packed struct {
    hand: u64, // 8 bytes - the 7-card hand
    rank: u16, // 2 bytes - the evaluation result
};

fn generateAllHands(allocator: std.mem.Allocator) ![]HandResult {
    const total_hands = 133784560; // C(52,7)

    // For testing, we could limit hands but for now generate all
    const max_hands = total_hands; // Generate all 133M hands
    var results = try allocator.alloc(HandResult, total_hands);
    var idx: usize = 0;

    // Generate all 7-card combinations from 52 cards
    // We need 7 nested loops to generate all C(52,7) combinations
    var c0: u6 = 0;
    while (c0 < 46) : (c0 += 1) {
        var c1: u6 = c0 + 1;
        while (c1 < 47) : (c1 += 1) {
            var c2: u6 = c1 + 1;
            while (c2 < 48) : (c2 += 1) {
                var c3: u6 = c2 + 1;
                while (c3 < 49) : (c3 += 1) {
                    var c4: u6 = c3 + 1;
                    while (c4 < 50) : (c4 += 1) {
                        var c5: u6 = c4 + 1;
                        while (c5 < 51) : (c5 += 1) {
                            var c6: u6 = c5 + 1;
                            while (c6 < 52) : (c6 += 1) {
                                // Create hand from 7 card indices
                                const hand = (@as(u64, 1) << c0) |
                                    (@as(u64, 1) << c1) |
                                    (@as(u64, 1) << c2) |
                                    (@as(u64, 1) << c3) |
                                    (@as(u64, 1) << c4) |
                                    (@as(u64, 1) << c5) |
                                    (@as(u64, 1) << c6);

                                // Evaluate using slow evaluator
                                const rank = poker.slow.evaluateHand(hand);

                                results[idx] = HandResult{
                                    .hand = hand,
                                    .rank = rank,
                                };
                                idx += 1;

                                // Check if we've reached the limit
                                if (idx >= max_hands) {
                                    std.debug.print("\nReached limit of {} hands\n", .{max_hands});
                                    return results[0..idx];
                                }

                                // Progress report every million hands
                                if (idx % 1000000 == 0) {
                                    const pct = @as(f64, @floatFromInt(idx)) / @as(f64, @floatFromInt(total_hands)) * 100.0;
                                    std.debug.print("\rProgress: {}/{} ({d:.1}%)", .{ idx, total_hands, pct });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std.debug.print("\nGenerated all {} hands\n", .{idx});
    return results[0..idx];
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    // Start timing
    var timer = try std.time.Timer.start();
    const start = timer.read();

    try stdout.print("Generating all 133,784,560 possible 7-card hands...\n", .{});
    try stdout.print("This will take approximately 2-3 minutes...\n\n", .{});

    // Generate all hands
    const results = try generateAllHands(allocator);
    defer allocator.free(results);

    const elapsed = timer.read() - start;
    const elapsed_sec = @as(f64, @floatFromInt(elapsed)) / 1e9;
    const hands_per_sec = @as(f64, @floatFromInt(results.len)) / elapsed_sec;

    try stdout.print("\nGeneration complete!\n", .{});
    try stdout.print("Time: {d:.2} seconds\n", .{elapsed_sec});
    try stdout.print("Speed: {d:.0} hands/second\n", .{hands_per_sec});
    try stdout.print("Total size: {} MB\n", .{results.len * @sizeOf(HandResult) / 1024 / 1024});

    // Write to file
    const file = try std.fs.cwd().createFile("all_hands.dat", .{});
    defer file.close();

    try stdout.print("\nWriting to all_hands.dat...\n", .{});

    // Write header
    const magic = "PKRH"; // Poker Hands
    const version: u32 = 1;
    const num_hands: u32 = @intCast(results.len);

    try file.writer().writeAll(magic);
    try file.writer().writeInt(u32, version, .little);
    try file.writer().writeInt(u32, num_hands, .little);

    // Write all results
    const bytes = std.mem.sliceAsBytes(results);
    try file.writer().writeAll(bytes);

    try stdout.print("File written successfully!\n", .{});

    // Compute some statistics
    var rank_counts = [_]u32{0} ** 7462;
    for (results) |result| {
        rank_counts[result.rank] += 1;
    }

    try stdout.print("\nRank distribution:\n", .{});
    // Show counts for major hand categories
    var royal_flush: u32 = 0;
    var straight_flush: u32 = 0;
    var four_kind: u32 = 0;
    var full_house: u32 = 0;
    var flush: u32 = 0;
    var straight: u32 = 0;
    var three_kind: u32 = 0;
    var two_pair: u32 = 0;
    var one_pair: u32 = 0;
    var high_card: u32 = 0;

    for (results) |result| {
        const category = poker.getHandCategory(result.rank);
        switch (category) {
            .straight_flush => if (result.rank <= 10) {
                if (result.rank == 0) royal_flush += 1 else straight_flush += 1;
            },
            .four_of_a_kind => four_kind += 1,
            .full_house => full_house += 1,
            .flush => flush += 1,
            .straight => straight += 1,
            .three_of_a_kind => three_kind += 1,
            .two_pair => two_pair += 1,
            .pair => one_pair += 1,
            .high_card => high_card += 1,
        }
    }

    // Known exact counts for 7-card poker hands
    // Source: https://en.wikipedia.org/wiki/Poker_probability#7-card_poker_hands
    // Also: "The Mathematics of Poker" by Bill Chen and Jerrod Ankenman
    const expected_counts = .{
        .royal_flush = 4324, // 4 suits × C(47,2) = 4 × 1081
        .straight_flush = 37260, // 9 ranks × 4 suits × C(46,2) = 36 × 1035
        .four_of_a_kind = 224848, // 13 × C(48,3) = 13 × 17296
        .full_house = 3473184, // C(13,2) × C(4,3)² × C(11,1) × 4 + 13 × C(4,3) × C(12,1) × C(4,2) × C(11,1) × 4
        .flush = 4047644, // [C(13,5) × 4 - 40] × C(39,2) - [C(13,7) × 4 - 4] × C(13,2) × 4²
        .straight = 6180020, // 10 × 4^5 × C(47,2) - 40 × C(46,2) - 36 × C(44,2)
        .three_of_a_kind = 6461620, // 13 × C(4,3) × C(12,4) × 4^4
        .two_pair = 31433400, // C(13,2) × C(4,2)² × C(11,1) × 4
        .one_pair = 58627800, // 13 × C(4,2) × C(12,5) × 4^5
        .high_card = 23294460, // [C(13,7) - 71] × (4^7 - 4)
    };

    try stdout.print("  Royal Flush:     {:>10} ({d:.6}%)\n", .{ royal_flush, @as(f64, @floatFromInt(royal_flush)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Straight Flush:  {:>10} ({d:.6}%)\n", .{ straight_flush, @as(f64, @floatFromInt(straight_flush)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Four of a Kind:  {:>10} ({d:.4}%)\n", .{ four_kind, @as(f64, @floatFromInt(four_kind)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Full House:      {:>10} ({d:.3}%)\n", .{ full_house, @as(f64, @floatFromInt(full_house)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Flush:           {:>10} ({d:.3}%)\n", .{ flush, @as(f64, @floatFromInt(flush)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Straight:        {:>10} ({d:.3}%)\n", .{ straight, @as(f64, @floatFromInt(straight)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Three of a Kind: {:>10} ({d:.2}%)\n", .{ three_kind, @as(f64, @floatFromInt(three_kind)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  Two Pair:        {:>10} ({d:.2}%)\n", .{ two_pair, @as(f64, @floatFromInt(two_pair)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  One Pair:        {:>10} ({d:.2}%)\n", .{ one_pair, @as(f64, @floatFromInt(one_pair)) / @as(f64, @floatFromInt(results.len)) * 100.0 });
    try stdout.print("  High Card:       {:>10} ({d:.2}%)\n", .{ high_card, @as(f64, @floatFromInt(high_card)) / @as(f64, @floatFromInt(results.len)) * 100.0 });

    const total = royal_flush + straight_flush + four_kind + full_house + flush + straight + three_kind + two_pair + one_pair + high_card;
    try stdout.print("  Total:           {:>10}\n", .{total});

    // Verify counts match expected values
    try stdout.print("\nVerifying against known values:\n", .{});

    if (royal_flush != expected_counts.royal_flush) {
        try stdout.print("  ❌ Royal Flush: expected {}, got {} (diff: {})\n", .{ expected_counts.royal_flush, royal_flush, @as(i32, @intCast(royal_flush)) - @as(i32, @intCast(expected_counts.royal_flush)) });
        return error.InvalidRoyalFlushCount;
    } else {
        try stdout.print("  ✅ Royal Flush count correct\n", .{});
    }

    if (straight_flush != expected_counts.straight_flush) {
        try stdout.print("  ❌ Straight Flush: expected {}, got {} (diff: {})\n", .{ expected_counts.straight_flush, straight_flush, @as(i32, @intCast(straight_flush)) - @as(i32, @intCast(expected_counts.straight_flush)) });
        return error.InvalidStraightFlushCount;
    } else {
        try stdout.print("  ✅ Straight Flush count correct\n", .{});
    }

    if (four_kind != expected_counts.four_of_a_kind) {
        try stdout.print("  ❌ Four of a Kind: expected {}, got {} (diff: {})\n", .{ expected_counts.four_of_a_kind, four_kind, @as(i32, @intCast(four_kind)) - @as(i32, @intCast(expected_counts.four_of_a_kind)) });
        return error.InvalidFourOfAKindCount;
    } else {
        try stdout.print("  ✅ Four of a Kind count correct\n", .{});
    }

    if (full_house != expected_counts.full_house) {
        try stdout.print("  ❌ Full House: expected {}, got {} (diff: {})\n", .{ expected_counts.full_house, full_house, @as(i32, @intCast(full_house)) - @as(i32, @intCast(expected_counts.full_house)) });
        return error.InvalidFullHouseCount;
    } else {
        try stdout.print("  ✅ Full House count correct\n", .{});
    }

    if (flush != expected_counts.flush) {
        try stdout.print("  ❌ Flush: expected {}, got {} (diff: {})\n", .{ expected_counts.flush, flush, @as(i32, @intCast(flush)) - @as(i32, @intCast(expected_counts.flush)) });
        return error.InvalidFlushCount;
    } else {
        try stdout.print("  ✅ Flush count correct\n", .{});
    }

    if (straight != expected_counts.straight) {
        try stdout.print("  ❌ Straight: expected {}, got {} (diff: {})\n", .{ expected_counts.straight, straight, @as(i32, @intCast(straight)) - @as(i32, @intCast(expected_counts.straight)) });
        return error.InvalidStraightCount;
    } else {
        try stdout.print("  ✅ Straight count correct\n", .{});
    }

    if (three_kind != expected_counts.three_of_a_kind) {
        try stdout.print("  ❌ Three of a Kind: expected {}, got {} (diff: {})\n", .{ expected_counts.three_of_a_kind, three_kind, @as(i32, @intCast(three_kind)) - @as(i32, @intCast(expected_counts.three_of_a_kind)) });
        return error.InvalidThreeOfAKindCount;
    } else {
        try stdout.print("  ✅ Three of a Kind count correct\n", .{});
    }

    if (two_pair != expected_counts.two_pair) {
        try stdout.print("  ❌ Two Pair: expected {}, got {} (diff: {})\n", .{ expected_counts.two_pair, two_pair, @as(i32, @intCast(two_pair)) - @as(i32, @intCast(expected_counts.two_pair)) });
        return error.InvalidTwoPairCount;
    } else {
        try stdout.print("  ✅ Two Pair count correct\n", .{});
    }

    if (one_pair != expected_counts.one_pair) {
        try stdout.print("  ❌ One Pair: expected {}, got {} (diff: {})\n", .{ expected_counts.one_pair, one_pair, @as(i32, @intCast(one_pair)) - @as(i32, @intCast(expected_counts.one_pair)) });
        return error.InvalidOnePairCount;
    } else {
        try stdout.print("  ✅ One Pair count correct\n", .{});
    }

    if (high_card != expected_counts.high_card) {
        try stdout.print("  ❌ High Card: expected {}, got {} (diff: {})\n", .{ expected_counts.high_card, high_card, @as(i32, @intCast(high_card)) - @as(i32, @intCast(expected_counts.high_card)) });
        return error.InvalidHighCardCount;
    } else {
        try stdout.print("  ✅ High Card count correct\n", .{});
    }

    try stdout.print("\n✅ ALL HAND COUNTS MATCH EXPECTED VALUES!\n", .{});
}
