const std = @import("std");
const poker = @import("poker");

const HandResult = packed struct {
    hand: u64,
    rank: u16,
};

// Pre-warm caches by evaluating a subset of hands
fn warmupCaches(results: []const HandResult) void {
    // Warm up by evaluating first 64K hands (or all if less)
    const warmup_hands = @min(65536, results.len);

    // Single hand warmup
    for (0..@min(1024, warmup_hands)) |i| {
        std.mem.doNotOptimizeAway(poker.evaluateHand(results[i].hand));
    }

    // Batch warmup
    var i: usize = 0;
    while (i + 4 <= warmup_hands) : (i += 4) {
        const batch = @Vector(4, u64){
            results[i].hand,
            results[i + 1].hand,
            results[i + 2].hand,
            results[i + 3].hand,
        };
        std.mem.doNotOptimizeAway(poker.evaluateBatch4(batch));
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const stdout = std.io.getStdOut().writer();

    // Open the data file
    const file = try std.fs.cwd().openFile("all_hands.dat", .{});
    defer file.close();

    // Read header
    var magic: [4]u8 = undefined;
    _ = try file.read(&magic);
    if (!std.mem.eql(u8, &magic, "PKRH")) {
        return error.InvalidMagic;
    }

    const version = try file.reader().readInt(u32, .little);
    if (version != 1) {
        return error.UnsupportedVersion;
    }

    const num_hands = try file.reader().readInt(u32, .little);
    try stdout.print("Loading {} hands from all_hands.dat...\n", .{num_hands});

    // Read all hand results into memory first
    const results = try allocator.alloc(HandResult, num_hands);
    defer allocator.free(results);

    const bytes = std.mem.sliceAsBytes(results);
    const bytes_read = try file.read(bytes);
    if (bytes_read != bytes.len) {
        return error.IncompleteRead;
    }

    try stdout.print("Loaded all hands into memory ({d:.1} MB)\n", .{@as(f64, @floatFromInt(bytes.len)) / (1024.0 * 1024.0)});

    // Pre-warm caches
    try stdout.print("Warming up caches...\n", .{});
    warmupCaches(results);

    // Start verification
    var timer = try std.time.Timer.start();
    const start = timer.read();

    var mismatches: u32 = 0;
    var batch_mismatches: u32 = 0;
    const batch_size = 32;

    // Verify in batches
    var i: usize = 0;
    while (i + batch_size <= num_hands) : (i += batch_size) {
        // Create batch
        var hands: [batch_size]u64 = undefined;
        var expected: [batch_size]u16 = undefined;
        for (0..batch_size) |j| {
            hands[j] = results[i + j].hand;
            expected[j] = results[i + j].rank;
        }

        // Evaluate batch
        const batch = @as(@Vector(batch_size, u64), hands);
        const batch_results = poker.evaluateBatch32(batch);

        // Check results
        for (0..batch_size) |j| {
            if (batch_results[j] != expected[j]) {
                batch_mismatches += 1;
                try stdout.print("Batch mismatch at {}: hand=0x{X}, expected={}, got={}\n", .{ i + j, hands[j], expected[j], batch_results[j] });
            }
        }

        // Progress report
        if (i % 10000000 == 0 and i > 0) {
            const pct = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_hands)) * 100.0;
            try stdout.print("Progress: {d:.1}% ({}/{})\n", .{ pct, i, num_hands });
        }
    }

    // Verify remaining hands individually
    while (i < num_hands) : (i += 1) {
        const result = poker.evaluateHand(results[i].hand);
        if (result != results[i].rank) {
            mismatches += 1;
            try stdout.print("Single mismatch at {}: hand=0x{X}, expected={}, got={}\n", .{ i, results[i].hand, results[i].rank, result });
        }
    }

    const elapsed = timer.read() - start;
    const elapsed_sec = @as(f64, @floatFromInt(elapsed)) / 1e9;
    const hands_per_sec = @as(f64, @floatFromInt(num_hands)) / elapsed_sec;

    try stdout.print("\nVerification complete!\n", .{});
    try stdout.print("Time: {d:.3} seconds (excluding I/O and warmup)\n", .{elapsed_sec});
    try stdout.print("Speed: {d:.0} hands/second\n", .{hands_per_sec});
    try stdout.print("Throughput: {d:.2} ns/hand\n", .{1e9 / hands_per_sec});

    if (mismatches == 0 and batch_mismatches == 0) {
        try stdout.print("\n✅ ALL {} HANDS VERIFIED CORRECTLY!\n", .{num_hands});
    } else {
        try stdout.print("\n❌ Found {} single mismatches and {} batch mismatches\n", .{ mismatches, batch_mismatches });
    }

    // Run multiple timing passes for more rigorous measurement
    if (mismatches == 0 and batch_mismatches == 0) {
        try stdout.print("\nRunning performance benchmark (5 passes)...\n", .{});

        var times: [5]f64 = undefined;
        for (0..5) |pass| {
            const pass_start = timer.read();

            // Just measure the batch evaluation performance
            var idx: usize = 0;
            while (idx + 32 <= num_hands) : (idx += 32) {
                var batch_array: [32]u64 = undefined;
                for (0..32) |j| {
                    batch_array[j] = results[idx + j].hand;
                }
                const batch = @as(@Vector(32, u64), batch_array);
                const batch_results = poker.evaluateBatch32(batch);
                std.mem.doNotOptimizeAway(batch_results);
            }

            const pass_elapsed = timer.read() - pass_start;
            times[pass] = @as(f64, @floatFromInt(pass_elapsed)) / 1e9;

            const pass_hands_per_sec = @as(f64, @floatFromInt(idx)) / times[pass];
            try stdout.print("  Pass {}: {d:.3}s, {d:.0} hands/sec\n", .{ pass + 1, times[pass], pass_hands_per_sec });
        }

        // Calculate statistics
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        const median_time = times[2];
        const median_hands_per_sec = @as(f64, @floatFromInt((num_hands / 32) * 32)) / median_time;

        try stdout.print("\nMedian performance: {d:.0} hands/second ({d:.2} ns/hand)\n", .{
            median_hands_per_sec,
            1e9 / median_hands_per_sec,
        });
    }
}
