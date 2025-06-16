const std = @import("std");
const print = std.debug.print;
const poker = @import("poker.zig");
const equity = @import("equity.zig");
const ranges = @import("ranges.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});

    print("\n=== Hand Evaluation Demo ===\n", .{});
    demoHandEvaluation();

    print("\n=== Equity Evaluator Demo ===\n", .{});
    try demoEquityEvaluator(allocator);

    print("\n=== Range Equity Demo ===\n", .{});
    try demoRangeEquity(allocator);
}

fn demoHandEvaluation() void {
    // Demo 1: Royal flush (from README example)
    const royal_flush = poker.createHand(&.{
        .{ .hearts, .ace }, // Hole card 1
        .{ .spades, .ace }, // Hole card 2
        .{ .hearts, .king }, // Flop
        .{ .hearts, .queen }, // Flop
        .{ .hearts, .jack }, // Flop
        .{ .hearts, .ten }, // Turn
        .{ .clubs, .two }, // River
    });
    print("Royal flush: {}\n", .{royal_flush.evaluate()});

    // Demo 2: Four of a kind
    const quads = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .ace },
        .{ .diamonds, .ace },
        .{ .clubs, .ace },
        .{ .hearts, .king },
        .{ .spades, .queen },
        .{ .diamonds, .jack },
    });
    print("Four aces: {}\n", .{quads.evaluate()});

    // Demo 3: Full house
    const full_house = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .ace },
        .{ .diamonds, .ace },
        .{ .clubs, .king },
        .{ .hearts, .king },
        .{ .spades, .queen },
        .{ .diamonds, .jack },
    });
    print("Aces full of kings: {}\n", .{full_house.evaluate()});

    // Demo 4: Straight
    const straight = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .king },
        .{ .diamonds, .queen },
        .{ .clubs, .jack },
        .{ .hearts, .ten },
        .{ .spades, .nine },
        .{ .diamonds, .two },
    });
    print("Ace-high straight: {}\n", .{straight.evaluate()});

    // Demo 5: High card
    const high_card = poker.createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .king },
        .{ .diamonds, .queen },
        .{ .clubs, .jack },
        .{ .hearts, .nine },
        .{ .spades, .seven },
        .{ .diamonds, .two },
    });
    print("Ace high: {}\n", .{high_card.evaluate()});
}

fn demoEquityEvaluator(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    // Demo 1: Preflop AA vs KK - using comptime parsing (no allocation)
    const aa_cards = poker.mustParseCards("AhAs");
    const aa = [2]poker.Card{ aa_cards[0], aa_cards[1] };

    const kk_cards = poker.mustParseCards("KdKc");
    const kk = [2]poker.Card{ kk_cards[0], kk_cards[1] };

    const preflop_result = try equity.monteCarlo(aa, kk, &.{}, 100000, rng, allocator);
    print("AA vs KK preflop: {d:.1}% equity\n", .{preflop_result.equity() * 100});

    // Demo 2: Postflop equity
    const board_cards = poker.mustParseCards("AdKh7s");
    const postflop_result = try equity.monteCarlo(aa, kk, &board_cards, 50000, rng, allocator);
    print("AA vs KK on Ad Kh 7s: {d:.1}% equity\n", .{postflop_result.equity() * 100});

    // Demo 3: Multi-way equity
    const qq_cards = poker.mustParseCards("QhQs");
    const qq = [2]poker.Card{ qq_cards[0], qq_cards[1] };
    var hands = [_][2]poker.Card{ aa, kk, qq };
    const multiway_results = try equity.multiway(&hands, &.{}, 50000, rng, allocator);
    defer allocator.free(multiway_results);

    print("3-way preflop (AA vs KK vs QQ):\n", .{});
    print("  AA: {d:.1}%\n", .{multiway_results[0].equity() * 100});
    print("  KK: {d:.1}%\n", .{multiway_results[1].equity() * 100});
    print("  QQ: {d:.1}%\n", .{multiway_results[2].equity() * 100});
}

fn demoRangeEquity(allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(456);
    const rng = prng.random();

    // Create ranges using new parseRange() function
    var hero_range = try ranges.parseRange("AA,KK,QQ,JJ,AKs,AQs", allocator);
    defer hero_range.deinit();

    var villain_range = try ranges.parseRange("TT,99,88,KQs,QJs", allocator);
    defer villain_range.deinit();

    print("Hero range: {} combinations (AA, KK, QQ, JJ, AKs, AQs)\n", .{hero_range.handCount()});
    print("Villain range: {} combinations (TT, 99, 88, KQs, QJs)\n", .{villain_range.handCount()});

    // Demo 1: Preflop range vs range equity
    const empty_board = [_]poker.Card{};

    const start_time = std.time.nanoTimestamp();
    const preflop_result = try ranges.calculateRangeEquityMonteCarlo(&hero_range, &villain_range, &empty_board, 5000, // Number of simulations
        rng, allocator);
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    print("Preflop range vs range equity:\n", .{});
    print("  Hero (premium range): {d:.1}%\n", .{preflop_result.hero_equity * 100.0});
    print("  Villain (calling range): {d:.1}%\n", .{preflop_result.villain_equity * 100.0});
    print("  Calculation time: {d:.1}ms ({} valid simulations)\n", .{ duration_ms, preflop_result.total_simulations });

    // Demo 2: Compare with individual hand equity for reference
    const aa_cards2 = poker.mustParseCards("AhAs");
    const aa2 = [2]poker.Card{ aa_cards2[0], aa_cards2[1] };

    const tt_cards = poker.mustParseCards("TcTd");
    const tt = [2]poker.Card{ tt_cards[0], tt_cards[1] };

    const individual_result = try equity.monteCarlo(aa2, tt, &empty_board, 2000, rng, allocator);
    print("For comparison - AA vs TT heads-up: {d:.1}% vs {d:.1}%\n", .{ individual_result.equity() * 100.0, (1.0 - individual_result.equity()) * 100.0 });
}
