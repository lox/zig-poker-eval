const std = @import("std");
const print = std.debug.print;
const poker = @import("poker");
const ansi = @import("ansi.zig");
const benchmark = @import("tools");
const cli_lib = @import("cli.zig");

// ============================================================================
// COMMAND DEFINITIONS & METADATA
// ============================================================================

const OutputFormat = enum {
    table,
    json,
};

// Self-contained command definitions with metadata and handlers
const EquityCommand = struct {
    const Options = struct {
        hand1: []const u8,
        hand2: []const u8,
        board: ?[]const u8 = null,
        sims: u32 = 10000,
        format: OutputFormat = .table,
        verbose: bool = false,
    };

    const meta = cli_lib.CommandMeta{
        .name = "equity",
        .description = "Calculate hand vs hand equity using Monte Carlo simulation",
        .usage = "poker-eval equity <hand1> <hand2> [options]",
        .examples = &.{
            "poker-eval equity \"AhAs\" \"KdKc\"",
            "poker-eval equity \"AhAs\" \"KdKc\" --board \"AdKh7s\"",
            "poker-eval equity \"AhAs\" \"KdKc\" --sims 50000 --verbose",
        },
    };

    const positional_fields = &[_][]const u8{ "hand1", "hand2" };

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "hand1")) return "First player's hole cards (e.g., \"AhAs\")";
        if (std.mem.eql(u8, field_name, "hand2")) return "Second player's hole cards (e.g., \"KdKc\")";
        if (std.mem.eql(u8, field_name, "board")) return "Community board cards (e.g., \"AdKh7s\")";
        if (std.mem.eql(u8, field_name, "sims")) return "Number of Monte Carlo simulations to run";
        if (std.mem.eql(u8, field_name, "format")) return "Output format: table or json";
        if (std.mem.eql(u8, field_name, "verbose")) return "Show detailed information and statistics";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        ansi.printBold("🎯 Equity Analysis\n", .{});
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

        // Parse hands - try specific cards first, then range notation
        var hand1_cards: [2]poker.Hand = undefined;
        var hand2_cards: [2]poker.Hand = undefined;
        const is_range_vs_range = false;
        _ = is_range_vs_range; // Future use for range vs range calculations

        // Try parsing as specific hole cards (e.g., "AhAs")
        if (opts.hand1.len == 4 and opts.hand2.len == 4) {
            hand1_cards = parseHoleCards(opts.hand1) catch |err| {
                print("Error parsing hand1 '{s}': {}\n", .{ opts.hand1, err });
                return;
            };

            hand2_cards = parseHoleCards(opts.hand2) catch |err| {
                print("Error parsing hand2 '{s}': {}\n", .{ opts.hand2, err });
                return;
            };
        } else {
            // Try range notation (AA, KK, AKs, etc.)
            print("Error: Range notation not yet implemented. Use specific cards like \"AhAs\" \"KdKc\"\n", .{});
            print("Coming soon: Range vs range equity calculations\n", .{});
            return;
        }

        // Parse board if provided
        var board_cards: []const poker.Hand = &.{};
        var board_hand: poker.Hand = 0;
        if (opts.board) |board_str| {
            board_hand = parseBoardCards(board_str) catch |err| {
                print("Error parsing board '{s}': {}\n", .{ board_str, err });
                return;
            };

            // Convert to slice for equity function
            var board_list = std.ArrayList(poker.Hand).init(allocator);
            defer board_list.deinit();

            // Extract individual cards from board_hand
            var i: u6 = 0;
            while (i < 52) : (i += 1) {
                const card_bit = @as(u64, 1) << i;
                if (board_hand & card_bit != 0) {
                    try board_list.append(card_bit);
                }
            }
            board_cards = try board_list.toOwnedSlice();
            defer allocator.free(board_cards);
        }

        // Check for card conflicts
        const all_cards = hand1_cards[0] | hand1_cards[1] | hand2_cards[0] | hand2_cards[1] | board_hand;
        const total_cards = poker.countCards(all_cards);
        const expected_cards = 4 + poker.countCards(board_hand);

        if (total_cards != expected_cards) {
            print("Error: Duplicate cards detected\n", .{});
            return;
        }

        // Display setup
        print("Hand 1: {s}\n", .{opts.hand1});
        print("Hand 2: {s}\n", .{opts.hand2});
        if (opts.board) |board| {
            print("Board:  {s} ({} cards)\n", .{ board, poker.countCards(board_hand) });
        }
        print("Simulations: {}\n", .{opts.sims});

        // Run Monte Carlo simulation
        print("\nRunning simulation...\n", .{});

        var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const rng = prng.random();

        const result = poker.monteCarlo(hand1_cards, hand2_cards, board_cards, opts.sims, rng, allocator) catch |err| {
            print("Error running simulation: {}\n", .{err});
            return;
        };

        // Display results
        print("\n", .{});
        ansi.printBold("📊 Results\n", .{});

        switch (opts.format) {
            .table => {
                ansi.printGreen("Hand 1 equity: {d:.1}%\n", .{result.equity() * 100});
                ansi.printYellow("Hand 2 equity: {d:.1}%\n", .{(1.0 - result.equity()) * 100});

                if (opts.verbose) {
                    print("\nDetailed breakdown:\n", .{});
                    print("  Hand 1 wins: {} ({d:.1}%)\n", .{ result.wins, result.winRate() * 100 });
                    print("  Ties:        {} ({d:.1}%)\n", .{ result.ties, result.tieRate() * 100 });
                    print("  Hand 2 wins: {} ({d:.1}%)\n", .{ result.total_simulations - result.wins - result.ties, result.lossRate() * 100 });
                }
            },
            .json => {
                print("{{\"hand1_equity\": {d:.4}, \"hand2_equity\": {d:.4}, \"simulations\": {}}}\n", .{ result.equity(), 1.0 - result.equity(), result.total_simulations });
            },
        }
    }
};

const EvalCommand = struct {
    const Options = struct {
        cards: []const u8,
        board: ?[]const u8 = null,
        format: OutputFormat = .table,
        verbose: bool = false,
    };

    const meta = cli_lib.CommandMeta{
        .name = "eval",
        .description = "Evaluate 7-card poker hand strength and ranking",
        .usage = "poker-eval eval <cards> [options]",
        .examples = &.{
            "poker-eval eval \"AhAsKhQsJhThTc\"",
            "poker-eval eval \"AhAs\" --board \"KhQsJhThTc\"",
            "poker-eval eval \"AhAs\" --board \"KhQsJhThTc\" --verbose",
        },
    };

    const positional_fields = &[_][]const u8{"cards"};

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "cards")) return "7-card hand or hole cards (e.g., \"AhAsKhQsJhThTc\")";
        if (std.mem.eql(u8, field_name, "board")) return "Community board cards (e.g., \"AdKh7s\")";
        if (std.mem.eql(u8, field_name, "format")) return "Output format: table or json";
        if (std.mem.eql(u8, field_name, "verbose")) return "Show detailed information and statistics";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        _ = allocator;

        ansi.printBold("🃏 Hand Evaluation\n", .{});
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

        const cards_str = opts.cards;

        // Try to parse as either 7-card hand or hole cards + board
        var hand: poker.Hand = undefined;
        var is_hole_and_board = false;

        if (cards_str.len == 14) {
            // 7-card hand format
            hand = parse7CardHand(cards_str) catch |err| {
                print("Error parsing 7-card hand '{s}': {}\n", .{ cards_str, err });
                return;
            };
        } else if (cards_str.len == 4 and opts.board != null) {
            // Hole cards + board format
            const hole_cards = parseHoleCards(cards_str) catch |err| {
                print("Error parsing hole cards '{s}': {}\n", .{ cards_str, err });
                return;
            };

            const board = parseBoardCards(opts.board.?) catch |err| {
                print("Error parsing board '{s}': {}\n", .{ opts.board.?, err });
                return;
            };

            hand = hole_cards[0] | hole_cards[1] | board;
            is_hole_and_board = true;
        } else {
            print("Error: Invalid format. Use 7 cards (14 chars) or 2 hole cards + --board option\n", .{});
            return;
        }

        // Verify we have exactly 7 cards
        if (poker.countCards(hand) != 7) {
            print("Error: Hand must contain exactly 7 cards (found {})\n", .{poker.countCards(hand)});
            return;
        }

        // Evaluate the hand
        const rank = poker.evaluateHand(hand);
        const category = poker.getHandCategory(rank);

        // Display results
        if (is_hole_and_board) {
            print("Hole cards: {s}\n", .{cards_str});
            print("Board:      {s}\n", .{opts.board.?});
        } else {
            print("Hand: {s}\n", .{cards_str});
        }

        print("\n", .{});
        ansi.printGreen("Hand rank:     {}\n", .{rank});
        ansi.printGreen("Hand category: {}\n", .{category});

        // Show relative strength (lower rank = stronger)
        const strength_pct = (7462.0 - @as(f64, @floatFromInt(rank))) / 7462.0 * 100.0;
        ansi.printCyan("Strength:      {d:.1}% (stronger than {d:.1}% of hands)\n", .{ strength_pct, strength_pct });

        if (opts.verbose) {
            print("\nHand bits: 0x{X}\n", .{hand});
        }
    }
};

const RangeCommand = struct {
    const Options = struct {
        range: []const u8,
        format: OutputFormat = .table,
    };

    const meta = cli_lib.CommandMeta{
        .name = "range",
        .description = "Parse and analyze poker hand ranges",
        .usage = "poker-eval range <range> [options]",
        .examples = &.{
            "poker-eval range \"AA,KK,AKs\"",
            "poker-eval range \"22+,A2s+\" --format json",
        },
    };

    const positional_fields = &[_][]const u8{"range"};

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "range")) return "Hand range notation (e.g., \"AA,KK,AKs\")";
        if (std.mem.eql(u8, field_name, "format")) return "Output format: table or json";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        const range_str = opts.range;

        // Try to parse with our poker module
        var range = poker.parseRange(range_str, allocator) catch |err| {
            print("Error parsing range '{s}': {}\n", .{ range_str, err });
            return;
        };
        defer range.deinit();

        switch (opts.format) {
            .table => {
                ansi.printBold("📋 Range Analysis\n", .{});
                print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
                ansi.printCyan("Range: {s}\n", .{range_str});
                ansi.printGreen("Combinations: {}\n", .{range.handCount()});
            },
            .json => {
                print("{{\"range\": \"{s}\", \"combinations\": {}}}\n", .{ range_str, range.handCount() });
            },
        }
    }
};

const BenchCommand = struct {
    const Options = struct {
        iterations: u32 = 100000,
        quick: bool = false,
        warmup: bool = true,
        measure_overhead: bool = true,
        multiple_runs: bool = true,
        validate: bool = false,
        run_test: bool = false,
        test_hand: ?u64 = null,
        verbose: bool = false,
        show_comparison: bool = false,
    };

    const meta = cli_lib.CommandMeta{
        .name = "bench",
        .description = "Run performance benchmarks on the hand evaluator",
        .usage = "poker-eval bench [options]",
        .examples = &.{
            "poker-eval bench",
            "poker-eval bench --quick",
            "poker-eval bench --iterations 50000 --validate",
            "poker-eval bench --test_hand 0x1F00000000000",
        },
    };

    const positional_fields = &[_][]const u8{};

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "iterations")) return "Number of benchmark iterations";
        if (std.mem.eql(u8, field_name, "quick")) return "Skip warmup, overhead measurement, and multiple runs";
        if (std.mem.eql(u8, field_name, "warmup")) return "Enable cache warmup before benchmarking";
        if (std.mem.eql(u8, field_name, "measure_overhead")) return "Measure and subtract framework overhead";
        if (std.mem.eql(u8, field_name, "multiple_runs")) return "Run multiple times for statistical analysis";
        if (std.mem.eql(u8, field_name, "validate")) return "Validate results against reference implementation";
        if (std.mem.eql(u8, field_name, "run_test")) return "Run evaluator functionality test";
        if (std.mem.eql(u8, field_name, "test_hand")) return "Test specific hand (hex format, e.g., 0x1F00000000000)";
        if (std.mem.eql(u8, field_name, "verbose")) return "Show detailed information and statistics";
        if (std.mem.eql(u8, field_name, "show_comparison")) return "Show SIMD vs scalar performance comparison";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        ansi.printBold("🚀 Performance Benchmark\n", .{});
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

        // Convert our options to benchmark options
        var bench_options = benchmark.BenchmarkOptions{
            .iterations = opts.iterations,
            .warmup = opts.warmup,
            .measure_overhead = opts.measure_overhead,
            .multiple_runs = opts.multiple_runs,
            .verbose = opts.verbose,
        };

        // Handle quick mode
        if (opts.quick) {
            bench_options.multiple_runs = false;
            bench_options.warmup = false;
            bench_options.measure_overhead = false;
        }

        // Run specific test if requested
        if (opts.test_hand) |hand| {
            ansi.printBold("\n🔍 Single Hand Test\n", .{});
            const result = benchmark.testSingleHand(hand);
            print("Test hand:         0x{X}\n", .{hand});
            print("Slow evaluator:    {}\n", .{result.slow});
            print("Fast evaluator:    {}\n", .{result.fast});
            print("Match:             {s}\n", .{if (result.match) "✓" else "✗"});
            print("\n", .{});
        }

        // Run evaluator test if requested
        if (opts.run_test) {
            ansi.printBold("\n🧪 Evaluator Test\n", .{});
            benchmark.testEvaluator() catch |err| {
                ansi.printRed("❌ Evaluator test failed: {}\n", .{err});
                return;
            };
            ansi.printGreen("✅ Evaluator test passed\n", .{});
            print("\n", .{});
        }

        // Run main benchmark
        print("Running benchmark with {} iterations...\n", .{bench_options.iterations});
        if (bench_options.warmup) print("  • Cache warmup enabled\n", .{});
        if (bench_options.measure_overhead) print("  • Overhead measurement enabled\n", .{});
        if (bench_options.multiple_runs) print("  • Multiple runs for statistical analysis\n", .{});

        const result = try benchmark.runBenchmark(bench_options, allocator);

        // Display results
        print("\n", .{});
        ansi.printBold("📊 Benchmark Results\n", .{});
        print("  Total hands:         {}\n", .{result.total_hands});
        if (bench_options.measure_overhead) {
            print("  Framework overhead:  {d:.2} ns/hand\n", .{result.overhead_ns});
        }
        ansi.printGreen("  Batch performance:   {d:.2} ns/hand\n", .{result.batch_ns_per_hand});
        ansi.printGreen("  Hands per second:    {}\n", .{result.hands_per_second});

        if (bench_options.multiple_runs) {
            if (result.coefficient_variation > 0.05) {
                ansi.printYellow("  Variation:           {d:.2}% (high - consider stable environment)\n", .{result.coefficient_variation * 100});
            } else {
                ansi.printGreen("  Variation:           {d:.2}% (low - reliable measurement)\n", .{result.coefficient_variation * 100});
            }
        }

        if (opts.show_comparison) {
            print("\n", .{});
            ansi.printBold("🔄 Performance Comparison\n", .{});
            ansi.printCyan("  Batch (4x SIMD):     {d:.2} ns/hand ({} hands/sec)\n", .{ result.batch_ns_per_hand, result.hands_per_second });
            ansi.printYellow("  Single hand:         {d:.2} ns/hand ({d:.0} hands/sec)\n", .{ result.single_ns_per_hand, 1e9 / result.single_ns_per_hand });
            ansi.printGreen("  SIMD Speedup:        {d:.2}x\n", .{result.simd_speedup});
        }

        // Run validation if requested
        if (opts.validate) {
            print("\n", .{});
            ansi.printBold("✅ Correctness Validation\n", .{});

            // Generate validation hands
            var prng = std.Random.DefaultPrng.init(123); // Different seed
            var rng = prng.random();
            const validation_hands = try allocator.alloc(u64, 16000);
            defer allocator.free(validation_hands);

            for (validation_hands) |*hand| {
                hand.* = poker.generateRandomHand(&rng);
            }

            _ = benchmark.validateCorrectness(validation_hands) catch |err| {
                ansi.printRed("❌ Validation failed: {}\n", .{err});
                return;
            };
            ansi.printGreen("✅ All evaluations match reference implementation\n", .{});
        }
    }
};

const DemoCommand = struct {
    const Options = void;

    const meta = cli_lib.CommandMeta{
        .name = "demo",
        .description = "Show evaluator capabilities and example usage",
        .usage = "poker-eval demo",
        .examples = &.{
            "poker-eval demo",
        },
    };

    const positional_fields = &[_][]const u8{};

    fn getFieldDescription(field_name: []const u8) []const u8 {
        _ = field_name;
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        _ = opts;
        _ = allocator;

        ansi.printBold("🎮 Zig Poker Evaluator Demo\n", .{});
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

        ansi.printBold("\n🃏 High-Performance Hand Evaluation\n", .{});

        // Show evaluator performance
        var prng = std.Random.DefaultPrng.init(42);
        var rng = prng.random();

        const test_hands = [_]u64{
            0x1F00000000000, // Royal flush pattern
            poker.generateRandomHand(&rng),
            poker.generateRandomHand(&rng),
        };

        for (test_hands, 0..) |hand, i| {
            const result = poker.evaluateHand(hand);
            ansi.printGreen("Hand {}: 0x{X} -> Rank: {}\n", .{ i + 1, hand, result });
        }

        ansi.printBold("\n🚀 Batch Processing\n", .{});
        const batch = poker.generateRandomHandBatch(&rng);
        const batch_results = poker.evaluateBatch4(batch);

        for (0..4) |i| {
            ansi.printCyan("Batch[{}]: 0x{X} -> Rank: {}\n", .{ i, batch[i], batch_results[i] });
        }

        ansi.printYellow("\nTry: poker-eval --help for more options\n", .{});
        ansi.printYellow("Range parsing: poker-eval range \"AA,KK,AKs\"\n", .{});
        ansi.printYellow("Benchmarks: zig build bench\n", .{});
    }
};

// Main command union - this is what the CLI library operates on
const Commands = union(enum) {
    equity: EquityCommand.Options,
    eval: EvalCommand.Options,
    range: RangeCommand.Options,
    bench: BenchCommand.Options,
    demo,
    help,

    // App metadata
    pub fn getAppMeta() cli_lib.AppMeta {
        return cli_lib.AppMeta{
            .name = "poker-eval",
            .description = "High-performance poker hand evaluator with analysis tools",
            .global_options = &.{
                .{ .flag = "--help, -h", .description = "Show help information" },
                .{ .flag = "--verbose", .description = "Verbose output (available in most commands)" },
                .{ .flag = "--format table|json", .description = "Output format (default: table)" },
            },
            .examples = &.{
                "poker-eval equity --help               # Show equity command help",
                "poker-eval eval \"AhAsKhQsJhThTc\"         # Evaluate 7-card hand",
                "poker-eval equity \"AhAs\" \"KdKc\"        # Hand vs hand equity",
                "poker-eval bench --quick               # Quick benchmark",
            },
        };
    }

    // Command metadata - now delegates to command structs
    pub fn getCommandMeta(tag: std.meta.Tag(Commands)) cli_lib.CommandMeta {
        return switch (tag) {
            .equity => EquityCommand.meta,
            .eval => EvalCommand.meta,
            .range => RangeCommand.meta,
            .bench => BenchCommand.meta,
            .demo => DemoCommand.meta,
            .help => cli_lib.CommandMeta{
                .name = "help",
                .description = "Show help information",
                .usage = "poker-eval help",
                .examples = &.{ "poker-eval help", "poker-eval --help" },
            },
        };
    }

    // Positional field definitions - now delegates to command structs
    pub fn getPositionalFields(comptime T: type) []const []const u8 {
        return switch (T) {
            EquityCommand.Options => EquityCommand.positional_fields,
            EvalCommand.Options => EvalCommand.positional_fields,
            RangeCommand.Options => RangeCommand.positional_fields,
            BenchCommand.Options => BenchCommand.positional_fields,
            else => &[_][]const u8{},
        };
    }

    // Field descriptions - now delegates to command structs
    pub fn getFieldDescription(field_name: []const u8) []const u8 {
        // We need to determine which command this field belongs to at runtime
        // For now, we'll check all commands (could be optimized)
        var desc = EquityCommand.getFieldDescription(field_name);
        if (!std.mem.eql(u8, desc, "No description available")) return desc;

        desc = EvalCommand.getFieldDescription(field_name);
        if (!std.mem.eql(u8, desc, "No description available")) return desc;

        desc = RangeCommand.getFieldDescription(field_name);
        if (!std.mem.eql(u8, desc, "No description available")) return desc;

        desc = BenchCommand.getFieldDescription(field_name);
        if (!std.mem.eql(u8, desc, "No description available")) return desc;

        return "No description available";
    }
};

// Create the CLI instance
const PokerCli = cli_lib.Cli(Commands);

// Helper function to print hand categories statistics
fn printHandCategories(categories: anytype) void {
    _ = categories;
    // Note: HandCategories struct needs to be checked in poker module
    // This is a placeholder - we'll need to adapt based on what's available
    print("  Hand category breakdown not yet available\n", .{});
}

fn formatCard(card: poker.Hand) [2]u8 {
    _ = card;
    // TODO: Implement card formatting when needed
    return [2]u8{ 'A', 'h' };
}

/// Parse hole cards from string (e.g., "AsKd" -> [As, Kd])
fn parseHoleCards(cards_str: []const u8) ![2]poker.Hand {
    if (cards_str.len != 4) {
        return error.InvalidHoleCardFormat;
    }

    const card1 = poker.parseCard(cards_str[0..2]) catch return error.InvalidCard1;
    const card2 = poker.parseCard(cards_str[2..4]) catch return error.InvalidCard2;
    return [2]poker.Hand{ card1, card2 };
}

/// Parse board cards from string (e.g., "AsKdQh" -> combined Hand)
fn parseBoardCards(board_str: []const u8) !poker.Hand {
    if (board_str.len % 2 != 0) {
        return error.InvalidBoardFormat;
    }

    var board: poker.Hand = 0;
    var i: usize = 0;
    while (i < board_str.len) : (i += 2) {
        const card = poker.parseCard(board_str[i .. i + 2]) catch return error.InvalidBoardCard;
        board |= card;
    }
    return board;
}

/// Parse a 7-card hand from string (e.g., "AsKdQhJsTs9h8c")
fn parse7CardHand(cards_str: []const u8) !poker.Hand {
    if (cards_str.len != 14) {
        return error.Invalid7CardFormat;
    }

    var hand: poker.Hand = 0;
    var i: usize = 0;
    while (i < cards_str.len) : (i += 2) {
        const card = poker.parseCard(cards_str[i .. i + 2]) catch return error.InvalidCard;
        hand |= card;
    }
    return hand;
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    const result = PokerCli.parseArgs(allocator, argv) catch |err| {
        print("Error: Invalid arguments ({any}). Use --help for usage.\n", .{err});
        return;
    };

    switch (result) {
        .help_main => PokerCli.printMainHelp(),
        .help_command => |tag| PokerCli.printCommandHelp(tag),
        .command => |command| switch (command) {
            .equity => |opts| try EquityCommand.handle(opts, allocator),
            .eval => |opts| try EvalCommand.handle(opts, allocator),
            .range => |opts| try RangeCommand.handle(opts, allocator),
            .bench => |opts| try BenchCommand.handle(opts, allocator),
            .demo => try DemoCommand.handle({}, allocator),
            .help => PokerCli.printMainHelp(),
        },
    }
}
