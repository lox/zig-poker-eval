const std = @import("std");

// Import all test modules to run them together
test {
    std.testing.refAllDecls(@import("evaluator/evaluator.zig"));
    std.testing.refAllDecls(@import("evaluator/slow_evaluator.zig"));
}