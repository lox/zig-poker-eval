const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    var pass: u32 = 0;
    var fail: u32 = 0;

    for (builtin.test_functions) |test_fn| {
        std.testing.allocator_instance = .{};
        
        test_fn.func() catch |err| switch (err) {
            error.SkipZigTest => {
                // Skip tests don't count as failures
                continue;
            },
            else => {
                fail += 1;
                std.debug.print("❌ FAIL: {s}\n", .{test_fn.name});
                continue;
            },
        };

        pass += 1;
        
        // Check for memory leaks
        if (std.testing.allocator_instance.deinit() == .leak) {
            std.debug.print("💧 LEAK: {s}\n", .{test_fn.name});
        }
    }

    const total = pass + fail;
    
    if (fail == 0) {
        std.debug.print("✅ All {d} tests passed\n", .{total});
    } else {
        std.debug.print("❌ {d}/{d} tests failed\n", .{ fail, total });
        std.process.exit(1);
    }
}