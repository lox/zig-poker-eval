const std = @import("std");
const print = std.debug.print;
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = std.process.argsAlloc(allocator) catch &[_][:0]u8{};
    defer std.process.argsFree(allocator, args);

    // Parse command line flags
    var run_eval = false;
    var run_equity = false;
    var run_threaded = false;

    // Default to running all if no args provided
    if (args.len == 1) {
        run_eval = true;
        run_equity = true;
        run_threaded = true;
    } else {
        // Parse individual flags
        for (args[1..]) |arg| {
            if (std.mem.eql(u8, arg, "--eval")) {
                run_eval = true;
            } else if (std.mem.eql(u8, arg, "--equity")) {
                run_equity = true;
            } else if (std.mem.eql(u8, arg, "--equityThreaded")) {
                run_threaded = true;
            } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
                print("Usage: benchmark [--eval] [--equity] [--equityThreaded] [--help]\n", .{});
                print("  --eval           Run hand evaluation benchmark\n", .{});
                print("  --equity         Run single-threaded equity benchmark\n", .{});
                print("  --equityThreaded Run multi-threaded equity benchmark\n", .{});
                print("  --help, -h       Show this help message\n", .{});
                print("\nIf no flags provided, runs all benchmarks.\n", .{});
                return;
            } else {
                print("Unknown argument: {s}\n", .{arg});
                print("Use --help for usage information.\n", .{});
                return;
            }
        }
    }

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});

    if (run_eval) {
        try benchmark.runEvaluatorBenchmark(allocator);
    }

    if (run_equity) {
        try benchmark.runEquityBenchmark(allocator);
    }

    if (run_threaded) {
        try benchmark.runEquityBenchmarkThreaded(allocator);
    }
}
