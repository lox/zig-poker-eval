// Experiment 27: Search for a perfect magic number for all 49,205 RPCs
//
// Goal: Find a single magic constant M and shift S such that:
//   (rpc * M) >> S → unique index for all RPCs
//
// This would eliminate the CHD displacement array, saving one memory access.
//
// Usage:
//   zig build-exe src/tools/find_perfect_magic.zig -O ReleaseFast
//   ./find_perfect_magic --threads 16 --timeout 43200

const std = @import("std");
const builtin = @import("builtin");

// Configuration
const TABLE_SIZE: u32 = 131072; // 2^17 (same as CHD)
const SHIFT: u6 = 47; // 64 - 17 = 47
const EXPECTED_RPCS: u32 = 49205;

// Killer heuristic configuration
const MAX_KILLERS: usize = 16;

const KillerPair = struct {
    rpc1: u32,
    rpc2: u32,
    count: u32,
};

const SearchStats = struct {
    attempts: std.atomic.Value(u64),
    start_time: i64,
    last_report: std.atomic.Value(i64),

    fn init() SearchStats {
        return .{
            .attempts = std.atomic.Value(u64).init(0),
            .start_time = std.time.milliTimestamp(),
            .last_report = std.atomic.Value(i64).init(std.time.milliTimestamp()),
        };
    }

    fn recordAttempt(self: *SearchStats) void {
        _ = self.attempts.fetchAdd(1, .monotonic);
    }

    fn maybeReport(self: *SearchStats, thread_id: usize) void {
        const now = std.time.milliTimestamp();
        const last = self.last_report.load(.monotonic);

        // Report every 10 seconds
        if (now - last > 10_000) {
            if (self.last_report.cmpxchgWeak(last, now, .monotonic, .monotonic) == null) {
                const attempts = self.attempts.load(.monotonic);
                const elapsed_sec = @as(f64, @floatFromInt(now - self.start_time)) / 1000.0;
                const rate = @as(f64, @floatFromInt(attempts)) / elapsed_sec;

                std.debug.print("[Thread {}] {d:.2} million attempts in {d:.1}s ({d:.1}M/s)\n", .{
                    thread_id,
                    @as(f64, @floatFromInt(attempts)) / 1_000_000.0,
                    elapsed_sec,
                    rate / 1_000_000.0,
                });
            }
        }
    }
};

const SearchContext = struct {
    rpcs: []const u32,
    result: std.atomic.Value(u64),
    found: std.atomic.Value(bool),
    stats: *SearchStats,
    timeout_ms: u64,

    fn init(rpcs: []const u32, stats: *SearchStats, timeout_ms: u64) SearchContext {
        return .{
            .rpcs = rpcs,
            .result = std.atomic.Value(u64).init(0),
            .found = std.atomic.Value(bool).init(false),
            .stats = stats,
            .timeout_ms = timeout_ms,
        };
    }
};

fn generateCandidate(rng: std.Random) u64 {
    // Generate candidates with good bit distribution
    // Prefer odd numbers (better for multiplication)
    var magic = rng.int(u64);
    magic |= 1; // Ensure odd
    return magic;
}

const TestResult = struct {
    success: bool,
    collided_rpc1: u32 = 0,
    collided_rpc2: u32 = 0,
};

fn testMagic(magic: u64, rpcs: []const u32, killers: []const KillerPair, occupied: []bool) TestResult {
    // Quick test: check killer pairs first
    for (killers) |killer| {
        const idx1 = @as(u32, @intCast((killer.rpc1 *% magic) >> SHIFT));
        const idx2 = @as(u32, @intCast((killer.rpc2 *% magic) >> SHIFT));
        if (idx1 == idx2) {
            return .{ .success = false, .collided_rpc1 = killer.rpc1, .collided_rpc2 = killer.rpc2 };
        }
    }

    // Full test: check all RPCs using a reusable bitset
    @memset(occupied, false);

    for (rpcs) |rpc| {
        const idx = @as(u32, @intCast((rpc *% magic) >> SHIFT));

        if (occupied[idx]) {
            return .{ .success = false, .collided_rpc1 = rpc, .collided_rpc2 = 0 };
        }

        occupied[idx] = true;
    }

    return .{ .success = true }; // Perfect magic found!
}

fn updateKillers(killers: []KillerPair, killer_count: *usize, rpc1: u32, rpc2: u32) void {
    // Check if this pair already exists
    var i: usize = 0;
    while (i < killer_count.*) : (i += 1) {
        if ((killers[i].rpc1 == rpc1 and killers[i].rpc2 == rpc2) or
            (killers[i].rpc1 == rpc2 and killers[i].rpc2 == rpc1))
        {
            killers[i].count += 1;
            return;
        }
    }

    // Add new killer if space
    if (killer_count.* < killers.len) {
        killers[killer_count.*] = .{
            .rpc1 = rpc1,
            .rpc2 = rpc2,
            .count = 1,
        };
        killer_count.* += 1;
    } else {
        // Replace least frequent killer
        var min_idx: usize = 0;
        var min_count: u32 = killers[0].count;
        for (killers, 0..) |killer, j| {
            if (killer.count < min_count) {
                min_count = killer.count;
                min_idx = j;
            }
        }
        killers[min_idx] = .{
            .rpc1 = rpc1,
            .rpc2 = rpc2,
            .count = 1,
        };
    }
}

fn searchThread(ctx: *SearchContext, thread_id: usize) void {
    const seed = @as(u64, @intCast(thread_id)) +% @as(u64, @bitCast(std.time.milliTimestamp()));
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var killers = [_]KillerPair{.{ .rpc1 = 0, .rpc2 = 0, .count = 0 }} ** MAX_KILLERS;
    var killer_count: usize = 0;

    // Reusable occupied bitset - avoids allocation per test!
    var occupied = [_]bool{false} ** TABLE_SIZE;

    const deadline = std.time.milliTimestamp() + @as(i64, @intCast(ctx.timeout_ms));

    while (std.time.milliTimestamp() < deadline) {
        // Check if another thread found a solution
        if (ctx.found.load(.acquire)) {
            return;
        }

        const magic = generateCandidate(rng);
        ctx.stats.recordAttempt();
        ctx.stats.maybeReport(thread_id);

        // Test with killer heuristic (pass reusable bitset)
        const result = testMagic(magic, ctx.rpcs, killers[0..killer_count], &occupied);

        if (result.success) {
            std.debug.print("\n🎉 Thread {} FOUND PERFECT MAGIC: 0x{X:0>16}\n", .{ thread_id, magic });
            ctx.result.store(magic, .release);
            ctx.found.store(true, .release);
            return;
        }

        // Update killer heuristic with collision info
        if (result.collided_rpc1 != 0 and result.collided_rpc2 != 0) {
            updateKillers(&killers, &killer_count, result.collided_rpc1, result.collided_rpc2);
        }

        // Sort killers every 256 attempts
        if (killer_count > 0 and @as(u32, @truncate(ctx.stats.attempts.load(.monotonic))) & 0xFF == 0) {
            std.sort.pdq(KillerPair, killers[0..killer_count], {}, struct {
                fn lessThan(_: void, a: KillerPair, b: KillerPair) bool {
                    return a.count > b.count;
                }
            }.lessThan);
        }
    }

    std.debug.print("Thread {} timed out after {} attempts\n", .{
        thread_id,
        ctx.stats.attempts.load(.monotonic),
    });
}

fn extractRPCs(allocator: std.mem.Allocator) ![]u32 {
    // Generate all possible 7-card non-flush hands and extract their RPCs
    std.debug.print("Extracting all unique RPCs...\n", .{});

    var rpc_set = std.AutoHashMap(u32, void).init(allocator);
    defer rpc_set.deinit();

    // Enumerate all 7-card combinations (simplified - just generate patterns)
    // For now, use a faster approach: generate all possible rank patterns
    var rank_counts = [_]u8{0} ** 13;
    try generateRankPatterns(&rank_counts, 0, 7, &rpc_set);

    std.debug.print("Found {} unique RPCs\n", .{rpc_set.count()});

    // Convert to array
    var rpcs = try allocator.alloc(u32, rpc_set.count());
    var iter = rpc_set.keyIterator();
    var i: usize = 0;
    while (iter.next()) |rpc| {
        rpcs[i] = rpc.*;
        i += 1;
    }

    return rpcs;
}

fn generateRankPatterns(
    rank_counts: *[13]u8,
    rank_idx: usize,
    cards_remaining: u8,
    rpc_set: *std.AutoHashMap(u32, void),
) !void {
    if (cards_remaining == 0) {
        // Convert rank_counts to RPC
        var rpc: u32 = 0;
        for (rank_counts) |count| {
            rpc = rpc * 5 + count;
        }
        try rpc_set.put(rpc, {});
        return;
    }

    if (rank_idx >= 13) return;

    // Try 0-4 cards of this rank (limited by cards_remaining)
    const max_count = @min(4, cards_remaining);
    for (0..max_count + 1) |count| {
        rank_counts[rank_idx] = @intCast(count);
        try generateRankPatterns(rank_counts, rank_idx + 1, cards_remaining - @as(u8, @intCast(count)), rpc_set);
    }
    rank_counts[rank_idx] = 0;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip(); // program name

    // Parse arguments
    var thread_count: usize = 16;
    var timeout_hours: u64 = 720; // 30 days default

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--threads")) {
            if (args.next()) |val| {
                thread_count = try std.fmt.parseInt(usize, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--timeout")) {
            if (args.next()) |val| {
                timeout_hours = try std.fmt.parseInt(u64, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--help")) {
            std.debug.print(
                \\Usage: find_perfect_magic [options]
                \\
                \\Options:
                \\  --threads N     Number of threads (default: 16)
                \\  --timeout N     Timeout in hours (default: 720 = 30 days)
                \\  --help          Show this help
                \\
                \\Experiment 27: Search for perfect magic number
                \\
                \\Goal: Find a single magic M such that (rpc * M) >> 47 gives
                \\      a unique 17-bit index for all 49,205 RPCs.
                \\
                \\This would eliminate the 8KB displacement array and save
                \\one memory access per lookup.
                \\
                \\This is a long-running search - could take hours, days, or weeks.
                \\Let it run on a server and check back later!
                \\
            , .{});
            return;
        }
    }

    std.debug.print(
        \\
        \\═══════════════════════════════════════════════════════════
        \\  Experiment 27: Perfect Magic Search
        \\═══════════════════════════════════════════════════════════
        \\
        \\Configuration:
        \\  Table size:     {d} slots (2^17)
        \\  Load factor:    ~37.5% (49,205 / 131,072)
        \\  Threads:        {d}
        \\  Timeout:        {d} hours
        \\  Shift:          {d} bits
        \\
        \\Strategy:
        \\  - Random search with killer heuristic
        \\  - Each thread searches independently
        \\  - First to find wins
        \\
        \\Expected search space: 2^40 to 2^50 attempts
        \\Estimated time: Unknown (could be minutes or days)
        \\
        \\Starting search...
        \\
    , .{ TABLE_SIZE, thread_count, timeout_hours, SHIFT });

    const rpcs = try extractRPCs(allocator);
    defer allocator.free(rpcs);

    std.debug.print("✓ Loaded {} RPCs (expected {})\n", .{ rpcs.len, EXPECTED_RPCS });

    if (rpcs.len != EXPECTED_RPCS) {
        std.debug.print("⚠️  Warning: RPC count mismatch!\n", .{});
    }

    var stats = SearchStats.init();
    var ctx = SearchContext.init(rpcs, &stats, timeout_hours * 3600 * 1000);

    // Spawn worker threads
    const threads = try allocator.alloc(std.Thread, thread_count);
    defer allocator.free(threads);

    for (threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, searchThread, .{ &ctx, i });
    }

    // Wait for completion
    for (threads) |thread| {
        thread.join();
    }

    std.debug.print("\n", .{});

    if (ctx.found.load(.acquire)) {
        const magic = ctx.result.load(.acquire);
        const total_attempts = stats.attempts.load(.monotonic);
        const elapsed = std.time.milliTimestamp() - stats.start_time;

        std.debug.print(
            \\
            \\═══════════════════════════════════════════════════════════
            \\  SUCCESS! 🎉
            \\═══════════════════════════════════════════════════════════
            \\
            \\Found perfect magic: 0x{X:0>16}
            \\
            \\Search statistics:
            \\  Total attempts:  {d}
            \\  Elapsed time:    {d:.1} hours
            \\  Search rate:     {d:.1} million/sec
            \\
            \\Next steps:
            \\  1. Verify correctness (run verification tool)
            \\  2. Benchmark vs CHD (compare lookup speed)
            \\  3. Update mphf.zig if speedup >20%
            \\
            \\Add to src/internal/mphf.zig:
            \\
            \\pub const PERFECT_MAGIC: u64 = 0x{X:0>16};
            \\
            \\pub inline fn lookupPerfect(rpc: u32, value_table: []const u16) u16 {{
            \\    const h = @as(u64, rpc) *% PERFECT_MAGIC;
            \\    const idx = @as(u32, @intCast(h >> {d}));
            \\    return value_table[idx];
            \\}}
            \\
        , .{
            magic,
            total_attempts,
            @as(f64, @floatFromInt(elapsed)) / 3600000.0,
            @as(f64, @floatFromInt(total_attempts)) / @as(f64, @floatFromInt(elapsed)),
            magic,
            SHIFT,
        });
    } else {
        const total_attempts = stats.attempts.load(.monotonic);
        const elapsed = std.time.milliTimestamp() - stats.start_time;

        std.debug.print(
            \\
            \\═══════════════════════════════════════════════════════════
            \\  TIMEOUT
            \\═══════════════════════════════════════════════════════════
            \\
            \\No perfect magic found in {d} hours
            \\
            \\Search statistics:
            \\  Total attempts:  {d}
            \\  Elapsed time:    {d:.1} hours
            \\  Search rate:     {d:.1} million/sec
            \\  Search space:    ~2^{d:.1}
            \\
            \\Conclusion:
            \\  - CHD was the right choice
            \\  - Two-level hashing more reliable than perfect magic
            \\  - Current 3.27ns performance is excellent
            \\
            \\Consider:
            \\  - Increase timeout (try 24-48 hours)
            \\  - Try different table sizes (2^18 = 50% load factor)
            \\  - Accept CHD's proven performance
            \\
        , .{
            timeout_hours,
            total_attempts,
            @as(f64, @floatFromInt(elapsed)) / 3600000.0,
            @as(f64, @floatFromInt(total_attempts)) / @as(f64, @floatFromInt(elapsed)),
            std.math.log2(@as(f64, @floatFromInt(total_attempts))),
        });
    }
}
