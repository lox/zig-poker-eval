const std = @import("std");
const profiler = @import("profiler.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const print = std.debug.print;
    print("=== ZIG POKER EVALUATOR PROFILING ===\n", .{});
    print("Apple Silicon M1 Performance Analysis\n\n", .{});

    // 1. Detailed component profiling
    try profiler.profileHandEvaluation(allocator);

    // 2. Instruction-level profiling
    try profiler.profileWithInstructionCounting(allocator);

    // 3. Memory access pattern analysis
    try profiler.profileMemoryAccess(allocator);

    print("\n=== PROFILING COMPLETE ===\n", .{});
}
