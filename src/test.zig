const std = @import("std");

// Comprehensive test runner - imports all test files directly
// This ensures all 60 tests are discovered and run
test {
    // Card tests
    _ = @import("card/mod.zig");

    // Evaluator tests
    _ = @import("evaluator/evaluator.zig");
    _ = @import("evaluator/slow_evaluator.zig");
    _ = @import("evaluator/build_tables.zig");

    // Import evaluator mod.zig to get any tests there
    _ = @import("evaluator/mod.zig");

    // Poker tests - these will work once we fix the imports
    _ = @import("poker/poker.zig");
    _ = @import("poker/equity.zig");
    _ = @import("poker/ranges.zig");
    _ = @import("poker/notation.zig");
    _ = @import("poker/simulation.zig");
}
