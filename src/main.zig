const std = @import("std");
const print = std.debug.print;
const poker = @import("poker.zig");
const equity = @import("equity.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});

    print("\n=== Hand Evaluation Demo ===\n", .{});
    demoHandEvaluation();

    print("\n=== Equity Evaluator Demo ===\n", .{});
    try demoEquityEvaluator(allocator);
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

    // Demo 1: Preflop AA vs KK
    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) };
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) };

    const preflop_result = try equity.equityMonteCarlo(aa, kk, &.{}, 100000, rng, allocator);
    print("AA vs KK preflop: {d:.1}% equity\n", .{preflop_result.equity() * 100});

    // Demo 2: Postflop equity
    const board = [_]poker.Card{
        poker.Card.init(14, 2), // Ad
        poker.Card.init(13, 0), // Kh
        poker.Card.init(7, 1), // 7s
    };
    const postflop_result = try equity.equityMonteCarlo(aa, kk, &board, 50000, rng, allocator);
    print("AA vs KK on Ad Kh 7s: {d:.1}% equity\n", .{postflop_result.equity() * 100});

    // Demo 3: Multi-way equity
    const qq = [_]poker.Card{ poker.Card.init(12, 0), poker.Card.init(12, 1) };
    var hands = [_][2]poker.Card{ aa, kk, qq };
    const multiway_results = try equity.equityMultiWayMonteCarlo(&hands, &.{}, 50000, rng, allocator);
    defer allocator.free(multiway_results);

    print("3-way preflop (AA vs KK vs QQ):\n", .{});
    print("  AA: {d:.1}%\n", .{multiway_results[0].equity() * 100});
    print("  KK: {d:.1}%\n", .{multiway_results[1].equity() * 100});
    print("  QQ: {d:.1}%\n", .{multiway_results[2].equity() * 100});
}
