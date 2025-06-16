const std = @import("std");
const poker = @import("poker.zig");
const equity = @import("equity.zig");
const equity_threaded = @import("equity_threaded.zig");
pub fn runEvaluatorBenchmark(allocator: std.mem.Allocator, json_output: bool) !void {
    const print = std.debug.print;

    if (!json_output) {
        print("=== Benchmark  ===\n", .{});
    }

    // Generate fixed set of hands using Hand API
    const hand_count = 10000;
    if (!json_output) {
        print("Generating {} random hands...\n", .{hand_count});
    }
    const hands = try allocator.alloc(poker.Hand, hand_count);
    defer allocator.free(hands);

    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (hands) |*hand| {
        hand.* = poker.generateRandomHand(random);
    }

    // Warm-up run
    var dummy: poker.HandRank = .high_card;
    for (hands) |hand| {
        const result = hand.evaluate();
        if (@intFromEnum(result) > @intFromEnum(dummy)) {
            dummy = result;
        }
    }

    // Main benchmark - multiple runs like Go's -count=3
    const runs = 3;
    var total_ops: u64 = 0;
    var total_ns: u64 = 0;

    for (0..runs) |run| {
        const start = std.time.nanoTimestamp();
        var ops: u64 = 0;
        const target_duration_ns = 1_000_000_000; // 1 second target

        // Run for approximately 1 second
        while (std.time.nanoTimestamp() - start < target_duration_ns) {
            for (hands) |hand| {
                const result = hand.evaluate();
                if (@intFromEnum(result) > @intFromEnum(dummy)) {
                    dummy = result;
                }
                ops += 1;
            }
        }

        const end = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end - start));
        const ns_per_op = duration_ns / ops;

        if (!json_output) {
            print("Run {}: {} ops, {d:.2} ns/op\n", .{ run + 1, ops, @as(f64, @floatFromInt(ns_per_op)) });
        }

        total_ops += ops;
        total_ns += duration_ns;
    }

    const avg_ns_per_op = total_ns / total_ops;
    const avg_ns_per_op_f64 = @as(f64, @floatFromInt(avg_ns_per_op));
    const evaluations_per_sec = 1_000_000_000.0 / avg_ns_per_op_f64;

    if (json_output) {
        const file = std.fs.cwd().createFile("bench_results.json", .{}) catch |err| {
            std.debug.print("Error creating results file: {}\n", .{err});
            return;
        };
        defer file.close();

        const json_content = std.fmt.allocPrint(allocator, "{{\"benchmark\":\"evaluator\",\"ns_per_op\":{d:.2},\"ops_per_sec\":{d:.0},\"runs\":{}}}\n", .{ avg_ns_per_op_f64, evaluations_per_sec, runs }) catch return;
        defer allocator.free(json_content);

        file.writeAll(json_content) catch |err| {
            std.debug.print("Error writing to results file: {}\n", .{err});
            return;
        };
    } else {
        print("\n=== Performance Summary ===\n", .{});
        print("{d:.2} ns/op (average across {} runs)\n", .{ avg_ns_per_op_f64, runs });
        print("{d:.1}M evaluations/second\n", .{evaluations_per_sec / 1_000_000.0});
    }
}

pub fn runEquityBenchmark(allocator: std.mem.Allocator, json_output: bool) !void {
    const print = std.debug.print;

    if (!json_output) {
        print("\n=== Equity Benchmark ===\n", .{});
    }

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Test scenarios - use createCard for individual cards
    const hero_hole = [2]poker.Card{ poker.createCard(.spades, .ace), poker.createCard(.diamonds, .king) };
    const villain_hole = [2]poker.Card{ poker.createCard(.hearts, .queen), poker.createCard(.clubs, .jack) };
    const board = [_]poker.Card{ poker.createCard(.hearts, .ace), poker.createCard(.spades, .seven), poker.createCard(.clubs, .two) };

    const simulations_per_run = 50000;
    const runs = 3;

    var total_simulations: u64 = 0;
    var total_ns: u64 = 0;

    for (0..runs) |run| {
        const start = std.time.nanoTimestamp();

        const result = try equity.equityMonteCarlo(hero_hole, villain_hole, &board, simulations_per_run, rng, allocator);

        const end = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end - start));
        const ns_per_sim = duration_ns / simulations_per_run;

        if (!json_output) {
            print("Run {}: {} sims, {d:.2} ns/sim, Hero equity: {d:.1}%\n", .{ run + 1, simulations_per_run, @as(f64, @floatFromInt(ns_per_sim)), result.equity() * 100.0 });
        }

        total_simulations += simulations_per_run;
        total_ns += duration_ns;
    }

    const avg_ns_per_sim = total_ns / total_simulations;
    const avg_ns_per_sim_f64 = @as(f64, @floatFromInt(avg_ns_per_sim));
    const simulations_per_sec = 1_000_000_000.0 / avg_ns_per_sim_f64;

    if (json_output) {
        const file = std.fs.cwd().createFile("bench_results.json", .{}) catch |err| {
            std.debug.print("Error creating results file: {}\n", .{err});
            return;
        };
        defer file.close();

        const json_content = std.fmt.allocPrint(allocator, "{{\"benchmark\":\"equity\",\"ns_per_sim\":{d:.2},\"sims_per_sec\":{d:.0},\"runs\":{}}}\n", .{ avg_ns_per_sim_f64, simulations_per_sec, runs }) catch return;
        defer allocator.free(json_content);

        file.writeAll(json_content) catch |err| {
            std.debug.print("Error writing to results file: {}\n", .{err});
            return;
        };
    } else {
        print("\n=== Equity Performance Summary ===\n", .{});
        print("{d:.2} ns/simulation (average across {} runs)\n", .{ avg_ns_per_sim_f64, runs });
        print("{d:.1}K simulations/second\n", .{simulations_per_sec / 1000.0});
    }
}

pub fn runEquityBenchmarkThreaded(allocator: std.mem.Allocator, json_output: bool) !void {
    const print = std.debug.print;

    if (!json_output) {
        print("\n=== Threaded Equity Benchmark ===\n", .{});
    }

    // Same test scenario as single-threaded
    const hero_hole = [2]poker.Card{ poker.createCard(.spades, .ace), poker.createCard(.diamonds, .king) };
    const villain_hole = [2]poker.Card{ poker.createCard(.hearts, .queen), poker.createCard(.clubs, .jack) };
    const board = [_]poker.Card{ poker.createCard(.hearts, .ace), poker.createCard(.spades, .seven), poker.createCard(.clubs, .two) };

    const simulations_per_run = 500000; // Larger batch for threading
    const runs = 3;
    const base_seed: u64 = 42;

    var total_simulations: u64 = 0;
    var total_ns: u64 = 0;

    for (0..runs) |run| {
        const start = std.time.nanoTimestamp();

        const result = try equity_threaded.equityMonteCarloThreaded(hero_hole, villain_hole, &board, simulations_per_run, base_seed, allocator);

        const end = std.time.nanoTimestamp();
        const duration_ns = @as(u64, @intCast(end - start));
        const ns_per_sim = duration_ns / simulations_per_run;

        if (!json_output) {
            print("Run {}: {} sims, {d:.2} ns/sim, Hero equity: {d:.1}%\n", .{ run + 1, simulations_per_run, @as(f64, @floatFromInt(ns_per_sim)), result.equity() * 100.0 });
        }

        total_simulations += simulations_per_run;
        total_ns += duration_ns;
    }

    const avg_ns_per_sim = total_ns / total_simulations;
    const avg_ns_per_sim_f64 = @as(f64, @floatFromInt(avg_ns_per_sim));
    const simulations_per_sec = 1_000_000_000.0 / avg_ns_per_sim_f64;

    if (json_output) {
        const file = std.fs.cwd().createFile("bench_results.json", .{}) catch |err| {
            std.debug.print("Error creating results file: {}\n", .{err});
            return;
        };
        defer file.close();

        const json_content = std.fmt.allocPrint(allocator, "{{\"benchmark\":\"equity_threaded\",\"ns_per_sim\":{d:.2},\"sims_per_sec\":{d:.0},\"runs\":{}}}\n", .{ avg_ns_per_sim_f64, simulations_per_sec, runs }) catch return;
        defer allocator.free(json_content);

        file.writeAll(json_content) catch |err| {
            std.debug.print("Error writing to results file: {}\n", .{err});
            return;
        };
    } else {
        print("\n=== Threaded Equity Performance Summary ===\n", .{});
        print("{d:.2} ns/simulation (average across {} runs)\n", .{ avg_ns_per_sim_f64, runs });
        print("{d:.1}K simulations/second\n", .{simulations_per_sec / 1000.0});
    }
}
