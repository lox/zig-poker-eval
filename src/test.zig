const std = @import("std");

test {
    // Import modules to run their tests
    _ = @import("card");
    _ = @import("evaluator");
    _ = @import("poker");
}
