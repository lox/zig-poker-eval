const std = @import("std");
const print = std.debug.print;
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = std.process.argsAlloc(allocator) catch &[_][:0]u8{};
    defer std.process.argsFree(allocator, args);

    const run_eval = args.len == 1 or std.mem.eql(u8, args[1], "eval") or std.mem.eql(u8, args[1], "both");
    const run_equity = args.len == 1 or std.mem.eql(u8, args[1], "equity") or std.mem.eql(u8, args[1], "both");
    const run_threaded = args.len == 1 or std.mem.eql(u8, args[1], "threaded") or std.mem.eql(u8, args[1], "both");

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
