const std = @import("std");
const print = std.debug.print;
const poker = @import("poker.zig");
const benchmark = @import("benchmark.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("=== Zig 7-Card Texas Hold'em Evaluator ===\n", .{});
    try benchmark.runEvaluatorBenchmark(allocator);
}

// Simple integration test for main functionality
test "main integration" {
    // Just verify the main components work together
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test that benchmark can be called
    _ = try benchmark.runComprehensiveBenchmark(allocator);
}
