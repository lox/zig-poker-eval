const std = @import("std");
const poker = @import("poker");

const HandResult = packed struct {
    hand: u64,
    rank: u16,
};

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
    try stdout.print("Verifying {} hands from all_hands.dat...\n", .{num_hands});

    // Read all hand results
    const results = try allocator.alloc(HandResult, num_hands);
    defer allocator.free(results);

    const bytes = std.mem.sliceAsBytes(results);
    _ = try file.read(bytes);

    // Start verification
    var timer = try std.time.Timer.start();
    const start = timer.read();

    var mismatches: u32 = 0;
    var batch_mismatches: u32 = 0;
    const batch_size = 8;

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
        const batch_results = poker.evaluateBatch8(batch);

        // Check results
        for (0..batch_size) |j| {
            if (batch_results[j] != expected[j]) {
                batch_mismatches += 1;
                try stdout.print("Batch mismatch at {}: hand=0x{X}, expected={}, got={}\n", .{ i + j, hands[j], expected[j], batch_results[j] });
            }
        }

        // Progress report
        if (i % 1000000 == 0 and i > 0) {
            const pct = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_hands)) * 100.0;
            std.debug.print("\rBatch verification: {}/{} ({d:.1}%)", .{ i, num_hands, pct });
        }
    }

    std.debug.print("\rBatch verification: {}/{} (100.0%)\n", .{ num_hands, num_hands });

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
    try stdout.print("Time: {d:.2} seconds\n", .{elapsed_sec});
    try stdout.print("Speed: {d:.0} hands/second\n", .{hands_per_sec});

    if (mismatches == 0 and batch_mismatches == 0) {
        try stdout.print("\n✅ ALL {} HANDS VERIFIED CORRECTLY!\n", .{num_hands});
    } else {
        try stdout.print("\n❌ Found {} single mismatches and {} batch mismatches\n", .{ mismatches, batch_mismatches });
    }

    // Also do a sampling test with random access
    try stdout.print("\nDoing random sampling test...\n", .{});
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();

    var sample_errors: u32 = 0;
    for (0..10000) |_| {
        const idx = rng.intRangeAtMost(usize, 0, num_hands - 1);
        const hand = results[idx].hand;
        const expected = results[idx].rank;
        const got = poker.evaluateHand(hand);

        if (got != expected) {
            sample_errors += 1;
            try stdout.print("Sample error at {}: hand=0x{X}, expected={}, got={}\n", .{ idx, hand, expected, got });
        }
    }

    if (sample_errors == 0) {
        try stdout.print("✅ Random sampling (10,000 hands) passed!\n", .{});
    } else {
        try stdout.print("❌ Random sampling found {} errors\n", .{sample_errors});
    }
}
