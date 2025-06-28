const std = @import("std");

// Test all modules - this file is designed to work with zig build test
// For direct testing without modules, use individual file tests
test {
    // Import all modules to run their tests
    _ = @import("card");
    _ = @import("evaluator");
    _ = @import("poker");
}

// Ensure all declarations are tested
test {
    std.testing.refAllDecls(@This());
}
