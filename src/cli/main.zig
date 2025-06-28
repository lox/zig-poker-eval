const std = @import("std");
const print = std.debug.print;
const evaluator = @import("evaluator");
const poker = @import("poker");
const ansi = @import("ansi.zig");
const benchmark = @import("tools");

// Helper function to print hand categories statistics
fn printHandCategories(categories: anytype) void {
    _ = categories;
    // Note: HandCategories struct needs to be checked in poker module
    // This is a placeholder - we'll need to adapt based on what's available
    print("  Hand category breakdown not yet available\n", .{});
}

const Command = enum { equity, eval, range, bench, demo, help };

fn formatCard(card: poker.Card) [2]u8 {
    _ = card;
    // Note: This needs to be adapted based on poker.Card API
    // Placeholder implementation
    return [2]u8{ 'A', 'h' };
}

/// Parse hand notation (e.g., "AKo", "88", "AhKs") into a specific 2-card hand
fn parseHandNotation(hand_str: []const u8, rng: std.Random, allocator: std.mem.Allocator) ![2]poker.Card {
    // Try unified notation parsing first
    if (poker.parse(hand_str, allocator)) |combinations| {
        defer allocator.free(combinations);
        if (combinations.len > 0) {
            const idx = rng.intRangeLessThan(usize, 0, combinations.len);
            return combinations[idx];
        }
    } else |_| {}

    // Fall back to specific card parsing - this needs to be implemented
    return error.ParsingNotYetImplemented;
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
    _ = allocator;
    if (config.args.len < 2) {
        print("Usage: poker-eval equity <hand1> <hand2> [--board <cards>] [--sims <count>]\n", .{});
        print("Example: poker-eval equity \"AhAs\" \"KdKc\" --board \"AdKh7s\"\n", .{});
        print("Note: Full equity analysis coming soon - poker module integration in progress\n", .{});
        return;
    }

    ansi.printBold("🎯 Equity Analysis (Demo)\n", .{});
    print("Hand 1: {s}\n", .{config.args[0]});
    print("Hand 2: {s}\n", .{config.args[1]});
    if (config.board) |board| print("Board: {s}\n", .{board});
    print("Simulations: {}\n", .{config.sims});
    print("\nFull implementation coming soon with poker module integration!\n", .{});
}

fn handleEval(config: Config, allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (config.args.len < 1) {
        print("Usage: poker-eval eval <7cards> OR <2cards> <5cards>\n", .{});
        print("Example: poker-eval eval \"AhAsKhQsJhThTc\"\n", .{});
        return;
    }

    ansi.printBold("🃏 Hand Evaluation (Demo)\n", .{});
    print("Cards: {s}\n", .{config.args[0]});
    print("\nFull hand evaluation coming soon with poker module integration!\n", .{});
    
    // Show basic evaluator functionality
    print("\nMeanwhile, here's our high-performance evaluator in action:\n", .{});
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const test_hand = evaluator.generateRandomHand(&rng);
    const result = evaluator.evaluateHand(test_hand);
    print("Random hand: 0x{X} -> Rank: {}\n", .{test_hand, result});
}

fn handleRange(config: Config, allocator: std.mem.Allocator) !void {
    if (config.args.len < 1) {
        print("Usage: poker-eval range <range>\n", .{});
        print("Example: poker-eval range \"AA,KK,AKs\"\n", .{});
        return;
    }

    const range_str = config.args[0];
    
    // Try to parse with our poker module
    var range = poker.parseRange(range_str, allocator) catch |err| {
        print("Error parsing range '{s}': {}\n", .{range_str, err});
        return;
    };
    defer range.deinit();

    switch (config.format) {
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

fn handleBench(config: Config, allocator: std.mem.Allocator) !void {
    ansi.printBold("🚀 Performance Benchmark\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    
    // Parse bench-specific arguments
    var bench_options = benchmark.BenchmarkOptions{};
    var run_validation = false;
    var run_test = false;
    var test_hand: ?u64 = null;
    
    // Parse additional arguments
    var i: usize = 0;
    while (i < config.args.len) {
        const arg = config.args[i];
        if (std.mem.eql(u8, arg, "--quick")) {
            bench_options.multiple_runs = false;
            bench_options.warmup = false;
            bench_options.measure_overhead = false;
        } else if (std.mem.eql(u8, arg, "--no-warmup")) {
            bench_options.warmup = false;
        } else if (std.mem.eql(u8, arg, "--no-overhead")) {
            bench_options.measure_overhead = false;
        } else if (std.mem.eql(u8, arg, "--single-run")) {
            bench_options.multiple_runs = false;
        } else if (std.mem.eql(u8, arg, "--validate")) {
            run_validation = true;
        } else if (std.mem.eql(u8, arg, "--test")) {
            run_test = true;
        } else if (std.mem.startsWith(u8, arg, "--iterations=")) {
            const iter_str = arg["--iterations=".len..];
            bench_options.iterations = std.fmt.parseInt(u32, iter_str, 10) catch bench_options.iterations;
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            // Space-separated format
            i += 1;
            if (i < config.args.len) {
                bench_options.iterations = std.fmt.parseInt(u32, config.args[i], 10) catch bench_options.iterations;
            }
        } else if (std.mem.startsWith(u8, arg, "--test-hand=")) {
            const hand_str = arg["--test-hand=".len..];
            test_hand = std.fmt.parseInt(u64, hand_str, 0) catch null;
        } else if (std.mem.eql(u8, arg, "--test-hand")) {
            // Space-separated format
            i += 1;
            if (i < config.args.len) {
                test_hand = std.fmt.parseInt(u64, config.args[i], 0) catch null;
            }
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            bench_options.verbose = true;
        }
        i += 1;
    }

    // Run specific test if requested
    if (test_hand) |hand| {
        ansi.printBold("\n🔍 Single Hand Test\n", .{});
        const result = benchmark.testSingleHand(hand);
        print("Test hand:         0x{X}\n", .{hand});
        print("Slow evaluator:    {}\n", .{result.slow});
        print("Fast evaluator:    {}\n", .{result.fast});
        print("Match:             {s}\n", .{if (result.match) "✓" else "✗"});
        print("\n", .{});
    }

    // Run evaluator test if requested
    if (run_test) {
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
    
    if (bench_options.show_comparison) {
        print("\n", .{});
        ansi.printBold("🔄 Performance Comparison\n", .{});
        ansi.printCyan("  Batch (4x SIMD):     {d:.2} ns/hand ({} hands/sec)\n", .{result.batch_ns_per_hand, result.hands_per_second});
        ansi.printYellow("  Single hand:         {d:.2} ns/hand ({d:.0} hands/sec)\n", .{result.single_ns_per_hand, 1e9 / result.single_ns_per_hand});
        ansi.printGreen("  SIMD Speedup:        {d:.2}x\n", .{result.simd_speedup});
    }

    // Run validation if requested
    if (run_validation) {
        print("\n", .{});
        ansi.printBold("✅ Correctness Validation\n", .{});
        
        // Generate validation hands
        var prng = std.Random.DefaultPrng.init(123); // Different seed
        var rng = prng.random();
        const validation_hands = try allocator.alloc(u64, 16000);
        defer allocator.free(validation_hands);
        
        for (validation_hands) |*hand| {
            hand.* = evaluator.generateRandomHand(&rng);
        }
        
        _ = benchmark.validateCorrectness(validation_hands) catch |err| {
            ansi.printRed("❌ Validation failed: {}\n", .{err});
            return;
        };
        ansi.printGreen("✅ All evaluations match reference implementation\n", .{});
    }
}

fn handleDemo(allocator: std.mem.Allocator) !void {
    _ = allocator;
    
    ansi.printBold("🎮 Zig Poker Evaluator Demo\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

    ansi.printBold("\n🃏 High-Performance Hand Evaluation\n", .{});
    
    // Show evaluator performance
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    
    const test_hands = [_]u64{
        0x1F00000000000, // Royal flush pattern
        evaluator.generateRandomHand(&rng),
        evaluator.generateRandomHand(&rng),
    };
    
    for (test_hands, 0..) |hand, i| {
        const result = evaluator.evaluateHand(hand);
        ansi.printGreen("Hand {}: 0x{X} -> Rank: {}\n", .{i + 1, hand, result});
    }
    
    ansi.printBold("\n🚀 Batch Processing\n", .{});
    const batch = evaluator.generateRandomHandBatch(&rng);
    const batch_results = evaluator.evaluateBatch4(batch);
    
    for (0..4) |i| {
        ansi.printCyan("Batch[{}]: 0x{X} -> Rank: {}\n", .{i, batch[i], batch_results[i]});
    }

    ansi.printYellow("\nTry: poker-eval --help for more options\n", .{});
    ansi.printYellow("Range parsing: poker-eval range \"AA,KK,AKs\"\n", .{});
    ansi.printYellow("Benchmarks: zig build bench\n", .{});
}

fn printHelp() void {
    ansi.printBold("🃏 poker-eval\n", .{});
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    print("High-performance poker hand evaluator with analysis tools\n", .{});

    ansi.printBold("\n📖 Usage:\n", .{});
    print("  poker-eval <command> [options] [args...]\n", .{});

    ansi.printBold("\n⚙️  Commands:\n", .{});
    ansi.printCyan("  equity", .{});
    print(" <hand1> <hand2>     Calculate hand vs hand equity (coming soon)\n", .{});
    ansi.printCyan("  eval", .{});
    print(" <cards>               Evaluate hand strength (coming soon)\n", .{});
    ansi.printCyan("  range", .{});
    print(" <range>              Parse and analyze hand range\n", .{});
    ansi.printCyan("  bench", .{});
    print(" [options]            Run performance benchmark\n", .{});
    ansi.printCyan("  demo", .{});
    print("                       Show evaluator capabilities\n", .{});
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

    ansi.printBold("\n🔧 Benchmark Options:\n", .{});
    ansi.printYellow("  --quick", .{});
    print("                    Fast benchmark (no warmup/overhead/multiple runs)\n", .{});
    ansi.printYellow("  --iterations=N", .{});
    print("            Number of iterations (default: 100000)\n", .{});
    ansi.printYellow("  --iterations N", .{});
    print("             Space-separated format also supported\n", .{});
    ansi.printYellow("  --validate", .{});
    print("                Validate correctness against reference\n", .{});
    ansi.printYellow("  --test", .{});
    print("                     Run evaluator functionality test\n", .{});
    ansi.printYellow("  --test-hand=0xHEX", .{});
    print("         Test specific hand (hex format)\n", .{});
    ansi.printYellow("  --test-hand 0xHEX", .{});
    print("          Space-separated format also supported\n", .{});
    ansi.printYellow("  --single-run", .{});
    print("              Single run instead of statistical analysis\n", .{});
    ansi.printYellow("  --no-warmup", .{});
    print("               Skip cache warmup\n", .{});
    ansi.printYellow("  --no-overhead", .{});
    print("             Skip overhead measurement\n", .{});

    ansi.printBold("\n💡 Examples:\n", .{});
    print("  poker-eval demo                          # Show capabilities\n", .{});
    print("  poker-eval range \"AA,KK,AKs\"              # Parse range\n", .{});
    print("  poker-eval bench                         # Full benchmark\n", .{});
    print("  poker-eval bench --quick                 # Quick benchmark\n", .{});
    print("  poker-eval bench --validate --test       # With validation\n", .{});
    print("  poker-eval bench --iterations 50000      # Custom iterations\n", .{});
    print("  poker-eval bench --test-hand 0x1F00      # Test specific hand\n", .{});

    ansi.printBold("\n🚀 Performance:\n", .{});
    print("  Current evaluator: ~7ns/hand (140M+ hands/sec)\n", .{});
    print("  Architecture: SIMD batching with perfect hash tables\n", .{});
    print("  Memory footprint: 120KB lookup tables\n", .{});
}
