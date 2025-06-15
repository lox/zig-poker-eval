const std = @import("std");
const print = std.debug.print;
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});
    try benchmark.runEvaluatorBenchmark(allocator);
}
