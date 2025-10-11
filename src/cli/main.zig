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
        exact: bool = false,
        format: OutputFormat = .table,
        verbose: bool = false,
    };

    const meta = cli_lib.CommandMeta{
        .name = "equity",
        .description = "Calculate hand vs hand or range vs range equity using Monte Carlo or exact calculation",
        .usage = "poker-eval equity <hand1> <hand2> [options]",
        .examples = &.{
            "poker-eval equity \"AhAs\" \"KdKc\"              # Specific hands (Monte Carlo)",
            "poker-eval equity \"AA\" \"KK\"                  # Range vs range (Monte Carlo)",
            "poker-eval equity \"AA\" \"KK\" --exact          # Exact calculation (enumerates all boards)",
            "poker-eval equity \"AhAs\" \"KdKc\" --board \"AdKh7s\"",
            "poker-eval equity \"AA\" \"KK\" --sims 50000 --verbose",
            "poker-eval equity \"AA\" \"KK\" --board \"7s8h9d2c\" --exact --verbose",
        },
    };

    const positional_fields = &[_][]const u8{ "hand1", "hand2" };

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "hand1")) return "First player's hole cards or range (e.g., \"AhAs\" or \"AA\")";
        if (std.mem.eql(u8, field_name, "hand2")) return "Second player's hole cards or range (e.g., \"KdKc\" or \"KK\")";
        if (std.mem.eql(u8, field_name, "board")) return "Community board cards (e.g., \"AdKh7s\")";
        if (std.mem.eql(u8, field_name, "sims")) return "Number of Monte Carlo simulations to run (ignored with --exact)";
        if (std.mem.eql(u8, field_name, "exact")) return "Calculate exact equity by enumerating all possible boards";
        if (std.mem.eql(u8, field_name, "format")) return "Output format: table or json";
        if (std.mem.eql(u8, field_name, "verbose")) return "Show detailed information and statistics";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        ansi.printBold("🎯 Equity Analysis\n", .{});
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

        // Parse both hands as ranges (supports both range notation and specific cards)
        var hero_range = poker.parseRange(opts.hand1, allocator) catch |err| {
            print("Error parsing hand '{s}': {}\n", .{ opts.hand1, err });
            return;
        };
        defer hero_range.deinit();

        var villain_range = poker.parseRange(opts.hand2, allocator) catch |err| {
            print("Error parsing hand '{s}': {}\n", .{ opts.hand2, err });
            return;
        };
        defer villain_range.deinit();

        // Parse board if provided
        var board_cards: []const poker.Hand = &.{};
        var board_cards_buf: [5]poker.Hand = undefined;
        var board_cards_count: usize = 0;
        var board_hand: poker.Hand = 0;
        if (opts.board) |board_str| {
            board_hand = parseBoardCards(board_str) catch |err| {
                print("Error parsing board '{s}': {}\n", .{ board_str, err });
                return;
            };

            // Extract individual cards from board_hand into fixed buffer
            var i: u6 = 0;
            while (i < 52) : (i += 1) {
                const card_bit = @as(u64, 1) << i;
                if (board_hand & card_bit != 0) {
                    if (board_cards_count >= 5) {
                        print("Error: Board has more than 5 cards\n", .{});
                        return;
                    }
                    board_cards_buf[board_cards_count] = card_bit;
                    board_cards_count += 1;
                }
            }
            board_cards = board_cards_buf[0..board_cards_count];
        }

        // Display setup
        print("Hand 1: {s} ({} combo{s})\n", .{ opts.hand1, hero_range.handCount(), if (hero_range.handCount() == 1) @as([]const u8, "") else "s" });
        print("Hand 2: {s} ({} combo{s})\n", .{ opts.hand2, villain_range.handCount(), if (villain_range.handCount() == 1) @as([]const u8, "") else "s" });
        if (opts.board) |board| {
            print("Board:  {s} ({} cards)\n", .{ board, poker.countCards(board_hand) });
        }
        if (opts.exact) {
            print("Mode:   Exact calculation (enumerating all boards)\n", .{});
        } else {
            print("Simulations: {}\n", .{opts.sims});
        }

        // Run equity calculation
        if (opts.exact) {
            print("\nCalculating exact equity...\n", .{});
        } else {
            print("\nRunning simulation...\n", .{});
        }

        if (opts.exact and opts.verbose) {
            // Use detailed exact calculation for verbose mode
            const detailed_result = hero_range.equityExactDetailed(&villain_range, board_cards, allocator) catch |err| {
                print("Error running exact calculation: {}\n", .{err});
                return;
            };

            // Display results
            print("\n", .{});
            ansi.printBold("📊 Results\n", .{});

            switch (opts.format) {
                .table => {
                    ansi.printGreen("Hand 1 equity: {d:.1}%\n", .{detailed_result.hero_equity * 100});
                    ansi.printYellow("Hand 2 equity: {d:.1}%\n", .{detailed_result.villain_equity * 100});

                    print("\nDetailed breakdown:\n", .{});
                    print("  Hand 1: {s} ({} combo{s})\n", .{ opts.hand1, hero_range.handCount(), if (hero_range.handCount() == 1) @as([]const u8, "") else "s" });
                    print("  Hand 2: {s} ({} combo{s})\n", .{ opts.hand2, villain_range.handCount(), if (villain_range.handCount() == 1) @as([]const u8, "") else "s" });
                    print("  Total boards: {}\n", .{detailed_result.total_simulations});
                    print("  Win rate:     {d:.2}%\n", .{detailed_result.winRate() * 100});
                    print("  Tie rate:     {d:.2}%\n", .{detailed_result.tieRate() * 100});
                    print("  Loss rate:    {d:.2}%\n", .{detailed_result.lossRate() * 100});

                    // Print hand category breakdowns
                    print("\nHand 1 categories:\n", .{});
                    printHandCategoryBreakdown(detailed_result.hero_categories);

                    print("\nHand 2 categories:\n", .{});
                    printHandCategoryBreakdown(detailed_result.villain_categories);
                },
                .json => {
                    print("{{\"hand1_equity\": {d:.4}, \"hand2_equity\": {d:.4}, \"boards\": {}, \"hand1_combos\": {}, \"hand2_combos\": {}}}\n", .{ detailed_result.hero_equity, detailed_result.villain_equity, detailed_result.total_simulations, hero_range.handCount(), villain_range.handCount() });
                },
            }
        } else {
            // Use standard calculation (exact or Monte Carlo)
            var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
            const rng = prng.random();

            const range_result = if (opts.exact)
                hero_range.equityExact(&villain_range, board_cards, allocator) catch |err| {
                    print("Error running exact calculation: {}\n", .{err});
                    return;
                }
            else
                hero_range.equityMonteCarlo(&villain_range, board_cards, opts.sims, rng, allocator) catch |err| {
                    print("Error running simulation: {}\n", .{err});
                    return;
                };

            // Display results
            print("\n", .{});
            ansi.printBold("📊 Results\n", .{});

            switch (opts.format) {
                .table => {
                    ansi.printGreen("Hand 1 equity: {d:.1}%\n", .{range_result.hero_equity * 100});
                    ansi.printYellow("Hand 2 equity: {d:.1}%\n", .{range_result.villain_equity * 100});

                    if (opts.verbose) {
                        print("\nDetailed breakdown:\n", .{});
                        print("  Hand 1: {s} ({} combo{s})\n", .{ opts.hand1, hero_range.handCount(), if (hero_range.handCount() == 1) @as([]const u8, "") else "s" });
                        print("  Hand 2: {s} ({} combo{s})\n", .{ opts.hand2, villain_range.handCount(), if (villain_range.handCount() == 1) @as([]const u8, "") else "s" });
                        if (opts.exact) {
                            print("  Total boards: {}\n", .{range_result.total_simulations});
                        } else {
                            print("  Valid simulations: {}\n", .{range_result.total_simulations});
                        }
                        print("  Win rate:     {d:.2}%\n", .{range_result.winRate() * 100});
                        print("  Tie rate:     {d:.2}%\n", .{range_result.tieRate() * 100});
                        print("  Loss rate:    {d:.2}%\n", .{range_result.lossRate() * 100});
                    }
                },
                .json => {
                    const label = if (opts.exact) "boards" else "simulations";
                    print("{{\"hand1_equity\": {d:.4}, \"hand2_equity\": {d:.4}, \"{s}\": {}, \"hand1_combos\": {}, \"hand2_combos\": {}}}\n", .{ range_result.hero_equity, range_result.villain_equity, label, range_result.total_simulations, hero_range.handCount(), villain_range.handCount() });
                },
            }
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
            const hole_cards_hand = poker.maybeParseHand(cards_str) catch |err| {
                print("Error parsing hole cards '{s}': {}\n", .{ cards_str, err });
                return;
            };

            const board = parseBoardCards(opts.board.?) catch |err| {
                print("Error parsing board '{s}': {}\n", .{ opts.board.?, err });
                return;
            };

            hand = hole_cards_hand | board;
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
        baseline: bool = false,
        filter: ?[]const u8 = null,
        threshold: f64 = 5.0,
    };

    const meta = cli_lib.CommandMeta{
        .name = "bench",
        .description = "Run performance benchmarks",
        .usage = "poker-eval bench [options]",
        .examples = &.{
            "poker-eval bench",
            "poker-eval bench baseline",
            "poker-eval bench --filter eval",
        },
    };

    const positional_fields = &[_][]const u8{};

    fn getFieldDescription(field_name: []const u8) []const u8 {
        if (std.mem.eql(u8, field_name, "baseline")) return "Save results as baseline";
        if (std.mem.eql(u8, field_name, "filter")) return "Filter to specific suite (eval, showdown)";
        if (std.mem.eql(u8, field_name, "threshold")) return "Regression threshold percentage (default 5.0)";
        return "No description available";
    }

    fn handle(opts: Options, allocator: std.mem.Allocator) !void {
        // Get the mode-specific baseline path
        const baseline_path = try benchmark.getBaselinePath(allocator);
        defer allocator.free(baseline_path);

        // Run benchmarks (use more runs if saving baseline)
        var result = try benchmark.runAllBenchmarks(allocator, opts.filter, opts.baseline);
        defer result.deinit();

        // Save as baseline if requested
        if (opts.baseline) {
            // Check for unstable benchmarks before saving
            var has_unstable = false;
            var suite_iter = result.suites.iterator();
            while (suite_iter.next()) |suite_entry| {
                var bench_iter = suite_entry.value_ptr.iterator();
                while (bench_iter.next()) |bench_entry| {
                    const metric = bench_entry.value_ptr.*;
                    if (metric.cv >= 0.05) {
                        has_unstable = true;
                        break;
                    }
                }
                if (has_unstable) break;
            }

            if (has_unstable) {
                print("\n", .{});
                ansi.printRed("❌ Cannot save baseline - unstable measurements detected\n", .{});
                return error.UnstableBenchmarks;
            }

            try benchmark.saveBaseline(result, baseline_path);
            print("\n", .{});
            ansi.printGreen("✅ Baseline saved to {s}\n", .{basename(baseline_path)});
            return;
        }

        // Load and compare with baseline
        var baseline = benchmark.loadBaseline(baseline_path, allocator) catch |err| {
            if (err == error.FileNotFound) {
                ansi.printYellow("ℹ️  No baseline found, expected {s}\n", .{basename(baseline_path)});
                return;
            }
            return err;
        };
        defer baseline.deinit();

        var comparison = try benchmark.compareResults(baseline, result, opts.threshold, allocator);
        defer comparison.deinit();

        benchmark.printComparisonResult(comparison);

        // Exit non-zero on regression
        if (!comparison.passed) {
            std.process.exit(1);
        }
    }

    fn basename(path: []const u8) []const u8 {
        var i = path.len;
        while (i > 0) {
            i -= 1;
            if (path[i] == '/' or path[i] == '\\') {
                return path[i + 1 ..];
            }
        }
        return path;
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
        const batch = poker.generateRandomHandBatch(4, &rng);
        const batch_results = poker.evaluateBatch(4, batch);

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

// Helper function to print hand category breakdown
fn printHandCategoryBreakdown(categories: poker.equity.HandCategories) void {
    if (categories.total == 0) {
        print("  No hands evaluated\n", .{});
        return;
    }

    if (categories.straight_flush > 0) {
        print("  Straight flush: {} ({d:.2}%)\n", .{ categories.straight_flush, categories.percentage(categories.straight_flush) });
    }
    if (categories.four_of_a_kind > 0) {
        print("  Four of a kind: {} ({d:.2}%)\n", .{ categories.four_of_a_kind, categories.percentage(categories.four_of_a_kind) });
    }
    if (categories.full_house > 0) {
        print("  Full house:     {} ({d:.2}%)\n", .{ categories.full_house, categories.percentage(categories.full_house) });
    }
    if (categories.flush > 0) {
        print("  Flush:          {} ({d:.2}%)\n", .{ categories.flush, categories.percentage(categories.flush) });
    }
    if (categories.straight > 0) {
        print("  Straight:       {} ({d:.2}%)\n", .{ categories.straight, categories.percentage(categories.straight) });
    }
    if (categories.three_of_a_kind > 0) {
        print("  Three of a kind: {} ({d:.2}%)\n", .{ categories.three_of_a_kind, categories.percentage(categories.three_of_a_kind) });
    }
    if (categories.two_pair > 0) {
        print("  Two pair:       {} ({d:.2}%)\n", .{ categories.two_pair, categories.percentage(categories.two_pair) });
    }
    if (categories.pair > 0) {
        print("  Pair:           {} ({d:.2}%)\n", .{ categories.pair, categories.percentage(categories.pair) });
    }
    if (categories.high_card > 0) {
        print("  High card:      {} ({d:.2}%)\n", .{ categories.high_card, categories.percentage(categories.high_card) });
    }
}

fn formatCard(card: poker.Hand) [2]u8 {
    return poker.formatCard(card);
}

/// Parse board cards from string (e.g., "AsKdQh" -> combined Hand)
fn parseBoardCards(board_str: []const u8) !poker.Hand {
    if (board_str.len % 2 != 0) {
        return error.InvalidBoardFormat;
    }

    const num_cards = board_str.len / 2;
    if (num_cards > 5) {
        return error.TooManyBoardCards;
    }

    var board: poker.Hand = 0;
    var i: usize = 0;
    while (i < board_str.len) : (i += 2) {
        const card = poker.maybeParseCard(board_str[i .. i + 2]) catch return error.InvalidBoardCard;
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
        const card = poker.maybeParseCard(cards_str[i .. i + 2]) catch return error.InvalidCard;
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
