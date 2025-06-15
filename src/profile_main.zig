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
    print("Next steps:\n", .{});
    print("1. Use 'instruments -t \"Time Profiler\" ./zig-out/bin/profile' for detailed CPU profiling\n", .{});
    print("2. Use 'instruments -t \"Allocations\" ./zig-out/bin/profile' for memory profiling\n", .{});
    print("3. Use 'perf record -g ./zig-out/bin/profile' on Linux for call graph analysis\n", .{});
}