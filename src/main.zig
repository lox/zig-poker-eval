const std = @import("std");
const print = std.debug.print;
const poker = @import("poker.zig");
const equity = @import("equity.zig");
const ranges = @import("ranges.zig");
const notation = @import("notation.zig");
const ansi = @import("ansi.zig");

const Command = enum { equity, eval, range, bench, demo, help };

fn formatCard(card: poker.Card) [2]u8 {
    const card_rank = card.getRank();
    const card_suit = card.getSuit();
    const rank_char: u8 = switch (card_rank) {
        2 => '2',
        3 => '3',
        4 => '4',
        5 => '5',
        6 => '6',
        7 => '7',
        8 => '8',
        9 => '9',
        10 => 'T',
        11 => 'J',
        12 => 'Q',
        13 => 'K',
        14 => 'A',
        else => '?',
    };
    const suit_char: u8 = switch (card_suit) {
        0 => 'h',
        1 => 's',
        2 => 'd',
        3 => 'c',
    };
    return [2]u8{ rank_char, suit_char };
}

/// Parse hand notation (e.g., "AKo", "88", "AhKs") into a specific 2-card hand
fn parseHandNotation(hand_str: []const u8, rng: std.Random, allocator: std.mem.Allocator) ![2]poker.Card {
    // Try notation parsing first (AKo, AKs, 88, etc.)
    if (notation.getRandomCombination(hand_str, rng, allocator) catch null) |hand| {
        return hand;
    }

    // Fall back to specific card parsing (AhKs)
    const cards = try poker.parseCards(hand_str, allocator);
    defer allocator.free(cards);
    if (cards.len != 2) {
        return error.InvalidHandLength;
    }
    return [2]poker.Card{ cards[0], cards[1] };
}

const OutputFormat = enum { table, json };

const Config = struct {
    command: Command,
    sims: u32 = 10000,
    board: ?[]const u8 = null,
    format: OutputFormat = .table,
    verbose: bool = false,
    args: [][:0]u8,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printHelp();
        return;
    }

    const config = parseArgs(args, allocator) catch |err| {
        print("Error: Invalid arguments ({any}). Use --help for usage.\n", .{err});
        return;
    };

    defer allocator.free(config.args);

    switch (config.command) {
        .equity => try handleEquity(config, allocator),
        .eval => try handleEval(config, allocator),
        .range => try handleRange(config, allocator),
        .bench => try handleBench(config, allocator),
        .demo => try handleDemo(allocator),
        .help => printHelp(),
    }
}

fn parseArgs(args: [][:0]u8, allocator: std.mem.Allocator) !Config {
    var config = Config{
        .command = .help,
        .args = &.{},
    };

    if (args.len < 2) return config;

    const command_str = args[1];
    config.command = std.meta.stringToEnum(Command, command_str) orelse {
        if (std.mem.eql(u8, command_str, "--help") or std.mem.eql(u8, command_str, "-h")) {
            return Config{ .command = .help, .args = &.{} };
        }
        return error.InvalidCommand;
    };

    var remaining_args = std.ArrayList([:0]u8).init(allocator);
    defer remaining_args.deinit();

    var i: usize = 2;
    while (i < args.len) {
        const arg = args[i];
        if (std.mem.startsWith(u8, arg, "--")) {
            if (std.mem.eql(u8, arg, "--board")) {
                i += 1;
                if (i < args.len) config.board = args[i];
            } else if (std.mem.eql(u8, arg, "--sims")) {
                i += 1;
                if (i < args.len) config.sims = std.fmt.parseInt(u32, args[i], 10) catch 10000;
            } else if (std.mem.eql(u8, arg, "--format")) {
                i += 1;
                if (i < args.len) config.format = std.meta.stringToEnum(OutputFormat, args[i]) orelse .table;
            } else if (std.mem.eql(u8, arg, "--verbose")) {
                config.verbose = true;
            } else {
                // Pass through unrecognized -- options to subcommands
                try remaining_args.append(arg);
            }
        } else {
            try remaining_args.append(arg);
        }
        i += 1;
    }

    config.args = try remaining_args.toOwnedSlice();
    return config;
}

fn handleEquity(config: Config, allocator: std.mem.Allocator) !void {
    if (config.args.len < 2) {
        print("Usage: zig-poker-eval equity <hand1> <hand2> [--board <cards>] [--sims <count>]\n", .{});
        print("Example: zig-poker-eval equity \"AhAs\" \"KdKc\" --board \"AdKh7s\"\n", .{});
        return;
    }

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.microTimestamp()));
    const rng = prng.random();

    const hand1_str = config.args[0];
    const hand2_str = config.args[1];

    // Check if these are ranges (contain commas) or individual hands
    const is_range1 = std.mem.indexOf(u8, hand1_str, ",") != null;
    const is_range2 = std.mem.indexOf(u8, hand2_str, ",") != null;

    if (is_range1 or is_range2) {
        try handleRangeEquity(hand1_str, hand2_str, config, allocator, rng);
    } else {
        try handleHandEquity(hand1_str, hand2_str, config, allocator, rng);
    }
}

fn handleHandEquity(hand1_str: []const u8, hand2_str: []const u8, config: Config, allocator: std.mem.Allocator, rng: std.Random) !void {
    const hand1 = parseHandNotation(hand1_str, rng, allocator) catch |err| {
        print("Error parsing hand 1 '{s}': {}\n", .{ hand1_str, err });
        return;
    };

    const hand2 = parseHandNotation(hand2_str, rng, allocator) catch |err| {
        print("Error parsing hand 2 '{s}': {}\n", .{ hand2_str, err });
        return;
    };

    const board_cards = if (config.board) |board_str|
        try poker.parseCards(board_str, allocator)
    else
        try allocator.alloc(poker.Card, 0);
    defer if (config.board != null) allocator.free(board_cards);

    const result = try equity.monteCarlo(hand1, hand2, board_cards, config.sims, rng, allocator);

    switch (config.format) {
        .table => {
            ansi.printBold("🎯 Hand Equity Analysis\n", .{});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            ansi.printGreen("Hand 1: {s} - {d:.1}%\n", .{ hand1_str, result.equity() * 100 });
            ansi.printRed("Hand 2: {s} - {d:.1}%\n", .{ hand2_str, (1.0 - result.equity()) * 100 });
            if (config.board) |board| ansi.printCyan("Board:  {s}\n", .{board});
            print("Simulations: {}\n", .{config.sims});
        },
        .json => {
            print("{{\"hand1\": \"{s}\", \"equity1\": {d:.3}, \"hand2\": \"{s}\", \"equity2\": {d:.3}, \"simulations\": {}}}\n", .{ hand1_str, result.equity(), hand2_str, 1.0 - result.equity(), config.sims });
        },
    }
}

fn handleRangeEquity(range1_str: []const u8, range2_str: []const u8, config: Config, allocator: std.mem.Allocator, rng: std.Random) !void {
    var range1 = try ranges.parseRange(range1_str, allocator);
    defer range1.deinit();

    var range2 = try ranges.parseRange(range2_str, allocator);
    defer range2.deinit();

    const board_cards = if (config.board) |board_str|
        try poker.parseCards(board_str, allocator)
    else
        try allocator.alloc(poker.Card, 0);
    defer if (config.board != null) allocator.free(board_cards);

    const result = try ranges.calculateRangeEquityMonteCarlo(&range1, &range2, board_cards, config.sims, rng, allocator);

    switch (config.format) {
        .table => {
            ansi.printBold("📊 Range vs Range Equity\n", .{});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            ansi.printGreen("Range 1: {s} ({} combos) - {d:.1}%\n", .{ range1_str, range1.handCount(), result.hero_equity * 100 });
            ansi.printRed("Range 2: {s} ({} combos) - {d:.1}%\n", .{ range2_str, range2.handCount(), result.villain_equity * 100 });
            if (config.board) |board| ansi.printCyan("Board:   {s}\n", .{board});
            print("Valid simulations: {}\n", .{result.total_simulations});
        },
        .json => {
            print("{{\"range1\": \"{s}\", \"equity1\": {d:.3}, \"range2\": \"{s}\", \"equity2\": {d:.3}, \"simulations\": {}}}\n", .{ range1_str, result.hero_equity, range2_str, result.villain_equity, result.total_simulations });
        },
    }
}

fn handleEval(config: Config, allocator: std.mem.Allocator) !void {
    if (config.args.len < 1) {
        print("Usage: zig-poker-eval eval <7cards> OR <2cards> <5cards>\n", .{});
        print("Example: zig-poker-eval eval \"AhAsKhQsJhThTc\"\n", .{});
        print("Example: zig-poker-eval eval \"AhAs\" \"KhQsJhThTc\"\n", .{});
        return;
    }

    var all_cards: [7]poker.Card = undefined;
    var card_count: usize = 0;

    for (config.args) |arg| {
        const cards = try poker.parseCards(arg, allocator);
        defer allocator.free(cards);
        for (cards) |card| {
            if (card_count >= 7) break;
            all_cards[card_count] = card;
            card_count += 1;
        }
    }

    if (card_count != 7) {
        print("Error: Need exactly 7 cards, got {}\n", .{card_count});
        return;
    }

    var hand = poker.Hand.init();
    for (all_cards) |card| {
        hand.addCard(card);
    }
    const rank = hand.evaluate();

    switch (config.format) {
        .table => {
            ansi.printBold("🃏 Hand Evaluation\n", .{});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            ansi.printCyan("Cards: ", .{});
            for (all_cards, 0..) |card, i| {
                if (i > 0) print(" ", .{});
                const formatted = formatCard(card);
                ansi.printYellow("{c}{c}", .{ formatted[0], formatted[1] });
            }
            print("\n", .{});
            ansi.printGreen("Rank:  {}\n", .{rank});
        },
        .json => {
            print("{{\"cards\": [", .{});
            for (all_cards, 0..) |card, i| {
                if (i > 0) print(", ", .{});
                const formatted = formatCard(card);
                print("\"{c}{c}\"", .{ formatted[0], formatted[1] });
            }
            print("], \"rank\": \"{}\"}}\n", .{rank});
        },
    }
}

fn handleRange(config: Config, allocator: std.mem.Allocator) !void {
    if (config.args.len < 1) {
        print("Usage: zig-poker-eval range <range>\n", .{});
        print("Example: zig-poker-eval range \"AA,KK,AKs\"\n", .{});
        return;
    }

    const range_str = config.args[0];
    var range = try ranges.parseRange(range_str, allocator);
    defer range.deinit();

    switch (config.format) {
        .table => {
            ansi.printBold("📋 Range Analysis\n", .{});
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
            ansi.printCyan("Range: {s}\n", .{range_str});
            ansi.printGreen("Combinations: {}\n", .{range.handCount()});
            if (config.verbose) {
                // TODO: List all hands in range
            }
        },
        .json => {
            print("{{\"range\": \"{s}\", \"combinations\": {}}}\n", .{ range_str, range.handCount() });
        },
    }
}

fn handleBench(config: Config, allocator: std.mem.Allocator) !void {
    const benchmark = @import("benchmark.zig");
    const json_output = config.format == .json;

    // Parse bench-specific arguments
    var run_eval = false;
    var run_equity = false;
    var run_threaded = false;

    // Default to running all if no specific bench args provided
    if (config.args.len == 0) {
        run_eval = true;
        run_equity = true;
        run_threaded = true;
    } else {
        // Parse individual flags
        for (config.args) |arg| {
            if (std.mem.eql(u8, arg, "--eval")) {
                run_eval = true;
            } else if (std.mem.eql(u8, arg, "--equity")) {
                run_equity = true;
            } else if (std.mem.eql(u8, arg, "--threaded")) {
                run_threaded = true;
            } else {
                print("Unknown bench argument: {s}\n", .{arg});
                print("Available options: --eval, --equity, --threaded\n", .{});
                return;
            }
        }
    }

    if (run_eval) {
        try benchmark.runEvaluatorBenchmark(allocator, json_output);
    }

    if (run_equity) {
        try benchmark.runEquityBenchmark(allocator, json_output);
    }

    if (run_threaded) {
        try benchmark.runEquityBenchmarkThreaded(allocator, json_output);
    }
}

fn handleDemo(allocator: std.mem.Allocator) !void {
    ansi.printBold("🎮 Zig Poker Evaluator Demo\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

    ansi.printBold("\n🃏 Hand Evaluation\n", .{});
    const royal_cards = try poker.parseCards("AhKhQhJhThAs2c", allocator);
    defer allocator.free(royal_cards);
    var royal_hand = poker.Hand.init();
    for (royal_cards) |card| {
        royal_hand.addCard(card);
    }
    ansi.printGreen("Royal flush: {}\n", .{royal_hand.evaluate()});

    ansi.printBold("\n🎯 Equity Calculation\n", .{});
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    const aa = [2]poker.Card{ poker.createCard(poker.Suit.hearts, poker.Rank.ace), poker.createCard(poker.Suit.spades, poker.Rank.ace) };
    const kk = [2]poker.Card{ poker.createCard(poker.Suit.diamonds, poker.Rank.king), poker.createCard(poker.Suit.clubs, poker.Rank.king) };

    const result = try equity.monteCarlo(aa, kk, &.{}, 50000, rng, allocator);
    ansi.printCyan("AA vs KK: ", .{});
    ansi.printGreen("{d:.1}%", .{result.equity() * 100});
    print(" vs ", .{});
    ansi.printRed("{d:.1}%\n", .{(1.0 - result.equity()) * 100});

    ansi.printYellow("\nTry: zig-poker-eval --help for more options\n", .{});
}

fn printHelp() void {
    ansi.printBold("🃏 zig-poker-eval\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    print("High-performance poker hand evaluator\n", .{});

    ansi.printBold("\n📖 Usage:\n", .{});
    print("  zig-poker-eval <command> [options] [args...]\n", .{});

    ansi.printBold("\n⚙️  Commands:\n", .{});
    ansi.printCyan("  equity", .{});
    print(" <hand1> <hand2>     Calculate hand vs hand equity\n", .{});
    ansi.printCyan("  equity", .{});
    print(" <range1> <range2>   Calculate range vs range equity\n", .{});
    ansi.printCyan("  eval", .{});
    print(" <cards>               Evaluate 7-card hand strength\n", .{});
    ansi.printCyan("  range", .{});
    print(" <range>              Parse and analyze hand range\n", .{});
    ansi.printCyan("  bench", .{});
    print(" [--eval|--equity|--threaded]  Run performance benchmarks\n", .{});
    ansi.printCyan("  demo", .{});
    print("                       Show quick demo\n", .{});
    ansi.printCyan("  help", .{});
    print("                       Show this help\n", .{});

    ansi.printBold("\n🔧 Options:\n", .{});
    ansi.printYellow("  --board", .{});
    print(" <cards>            Board cards (e.g., \"AdKh7s\")\n", .{});
    ansi.printYellow("  --sims", .{});
    print(" <count>             Number of simulations (default: 10000)\n", .{});
    ansi.printYellow("  --format", .{});
    print(" table|json        Output format (default: table)\n", .{});
    ansi.printYellow("  --verbose", .{});
    print("                  Verbose output\n", .{});

    ansi.printBold("\n💡 Examples:\n", .{});
    print("  zig-poker-eval equity \"AhAs\" \"KdKc\"        # Specific cards\n", .{});
    print("  zig-poker-eval equity \"AKo\" \"88\"           # Shorthand notation\n", .{});
    print("  zig-poker-eval equity \"AKs\" \"QQ\"           # Suited vs pair\n", .{});
    print("  zig-poker-eval equity \"AA,KK\" \"QQ,JJ\" --board \"AdKh7s\"\n", .{});
    print("  zig-poker-eval eval \"AhAsKhQsJhThTc\"\n", .{});
    print("  zig-poker-eval range \"AA-TT,AKs\" --verbose\n", .{});
    print("  zig-poker-eval bench --eval --equity\n", .{});
}
