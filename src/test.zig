const std = @import("std");

test {
    // Card tests
    _ = @import("card/mod.zig");

    // Evaluator tests - import the files that contain tests directly
    _ = @import("evaluator/evaluator.zig");
    _ = @import("evaluator/slow_evaluator.zig");
    _ = @import("evaluator/build_tables.zig");

    // Poker tests
    _ = @import("poker/poker.zig");
    _ = @import("poker/equity.zig");
    _ = @import("poker/ranges.zig");
    _ = @import("poker/notation.zig");
    _ = @import("poker/simulation.zig");
}
