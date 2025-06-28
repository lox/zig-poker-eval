const std = @import("std");
const builtin = @import("builtin");

const Status = enum { pass, fail, skip, leak };

const TestResult = struct {
    name: []const u8,
    status: Status,
    duration_ns: u64,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();
    var results = try std.ArrayList(TestResult).initCapacity(alloc, @import("builtin").test_functions.len);

    var pass: u32 = 0;
    var fail: u32 = 0;
    var skip: u32 = 0;
    var leak: u32 = 0;
    const total_start = std.time.nanoTimestamp();

    for (@import("builtin").test_functions) |test_fn| {
        std.testing.allocator_instance = .{};
        const start = std.time.nanoTimestamp();
        var status: Status = .pass;
        test_fn.func() catch |err| switch (err) {
            error.SkipZigTest => {
                status = .skip;
                skip += 1;
            },
            else => {
                status = .fail;
                fail += 1;
            },
        };
        const end = std.time.nanoTimestamp();
        if (status == .pass) pass += 1;
        // Check for memory leaks
        if (std.testing.allocator_instance.deinit() == .leak) {
            leak += 1;
            if (status == .pass) status = .leak;
        }
        try results.append(TestResult{
            .name = test_fn.name,
            .status = status,
            .duration_ns = @as(u64, @intCast(end - start)),
        });
    }

    const total = pass + fail + skip;
    const total_end = std.time.nanoTimestamp();
    const total_time = total_end - total_start;

    // Print table header  
    std.debug.print("\n{s:<50} {s:<8} {s:>10}\n", .{ "Test", "Status", "Time (ms)" });
    std.debug.print("{s}\n", .{"----------------------------------------------------------------------"});
    for (results.items) |res| {
        const status_str = switch (res.status) {
            .pass => "PASS",
            .fail => "FAIL",
            .skip => "SKIP",
            .leak => "LEAK",
        };
        // Truncate long test names to fit column width
        const display_name = if (res.name.len > 49) res.name[0..46] ++ "..." else res.name;
        std.debug.print("{s:<50} {s:<8} {d:>10.3}\n", .{ display_name, status_str, @as(f64, @floatFromInt(res.duration_ns)) / 1_000_000.0 });
    }
    std.debug.print("{s}\n", .{"----------------------------------------------------------------------"});
    std.debug.print("Total time: {d:.3} ms\n", .{@as(f64, @floatFromInt(total_time)) / 1_000_000.0});

    if (fail == 0) {
        std.debug.print("✅ All {d} tests passed ({d} skipped, {d} leaked)\n", .{ total, skip, leak });
    } else {
        std.debug.print("❌ {d}/{d} tests failed ({d} skipped, {d} leaked)\n", .{ fail, total, skip, leak });
        std.process.exit(1);
    }
}