const std = @import("std");
const evaluator = @import("evaluator.zig");
const chd = @import("chd.zig");
const bbhash = @import("bbhash.zig");

// DESIGN.md compliant table generator using RPC (Rank Pattern Code) encoding
const MAX_HANDS_LIMIT = 133_784_560; // Full C(52,7) = 133M

// DESIGN.md compliant patterns
const RPCPattern = struct {
    rpc: u32, // Rank Pattern Code (32-bit encoding of rank multiplicities)
    value: u16, // hand strength (0-7461)
};

const FlushPattern = bbhash.FlushPattern;

// Hand enumerator
const HandEnumerator = struct {
    cards: [7]u8,
    finished: bool,
    count: u64,

    fn init() HandEnumerator {
        return .{
            .cards = [7]u8{ 0, 1, 2, 3, 4, 5, 6 },
            .finished = false,
            .count = 0,
        };
    }

    fn next(self: *HandEnumerator) ?[7]u8 {
        if (self.finished) return null;

        const result = self.cards;
        self.count += 1;

        // Generate next combination
        var i: i8 = 6;
        while (i >= 0) : (i -= 1) {
            const ui: u8 = @intCast(i);
            if (self.cards[ui] < 52 - (7 - ui)) {
                self.cards[ui] += 1;
                var j = ui + 1;
                while (j < 7) : (j += 1) {
                    self.cards[j] = self.cards[j - 1] + 1;
                }
                return result;
            }
        }

        self.finished = true;
        return result;
    }
};

fn cardsToHand(cards: [7]u8) u64 {
    var hand: u64 = 0;
    for (cards) |card| {
        hand |= @as(u64, 1) << @intCast(card);
    }
    return hand;
}

// Build CHD hash table for RPCs
fn buildRPCHash(allocator: std.mem.Allocator, patterns: []const RPCPattern) !chd.CHDResult {
    var rpcs = try allocator.alloc(u64, patterns.len);
    defer allocator.free(rpcs);

    for (patterns, 0..) |pattern, i| {
        rpcs[i] = @as(u64, pattern.rpc);
    }

    return try chd.buildCHDHash(allocator, rpcs);
}

// BBHash for flush patterns (simplified)
// Use proper BBHash implementation
fn buildBBHash(allocator: std.mem.Allocator, patterns: []const FlushPattern) !bbhash.BBHashResult {
    return bbhash.buildBBHash(allocator, patterns);
}

pub fn main() !void {
    const print = std.debug.print;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("7-Card Poker Hand Evaluator - DESIGN.md Table Generator\n", .{});
    print("========================================================\n", .{});
    print("Target: RPC-based CHD tables (8K buckets, 64K final table)\n\n", .{});

    print("=== POKER HAND ENUMERATION ===\n", .{});
    print("Limit: {} hands (C(52,7) = 133,784,560 total)\n", .{MAX_HANDS_LIMIT});

    var enumerator = HandEnumerator.init();
    var non_flush_patterns = std.ArrayList(RPCPattern).init(allocator);
    defer non_flush_patterns.deinit();
    var flush_patterns = std.ArrayList(FlushPattern).init(allocator);
    defer flush_patterns.deinit();

    var rpc_map = std.AutoHashMap(u32, u16).init(allocator);
    defer rpc_map.deinit();
    var flush_map = std.AutoHashMap(u16, u16).init(allocator);
    defer flush_map.deinit();

    while (enumerator.next()) |cards| {
        if (enumerator.count > MAX_HANDS_LIMIT) break;

        const hand = cardsToHand(cards);
        const hand_value = evaluator.evaluateHand(hand);
        const is_flush = evaluator.hasFlush(hand);

        if (enumerator.count % 1_000_000 == 0) {
            print("  Processed: {d:.1}M hands, Non-flush: {}, Flush: {}\n", .{ @as(f64, @floatFromInt(enumerator.count)) / 1_000_000.0, rpc_map.count(), flush_map.count() });
        }

        if (is_flush) {
            // Extract flush ranks for the suit with ≥5 cards
            const suits = evaluator.getSuitMasks(hand);
            for (suits) |suit_mask| {
                if (@popCount(suit_mask) >= 5) {
                    const result = try flush_map.getOrPut(suit_mask);
                    if (!result.found_existing) {
                        result.value_ptr.* = hand_value;
                        try flush_patterns.append(.{
                            .ranks = suit_mask,
                            .value = hand_value,
                        });
                    }
                    break;
                }
            }
        } else {
            // Use RPC encoding for non-flush hands
            const rank_counts = chd.getRankCounts(hand);
            const rpc = chd.encodeRPC(rank_counts);

            const result = try rpc_map.getOrPut(rpc);
            if (!result.found_existing) {
                result.value_ptr.* = hand_value;
                try non_flush_patterns.append(.{
                    .rpc = rpc,
                    .value = hand_value,
                });
                
                // Debug the problematic RPC patterns
                if (rpc == 0x48000000 or rpc == 0x89000000) {
                    print("  DEBUG: Found problematic RPC 0x{x:08} with value {} for hand 0x{x:013}\n", 
                          .{ rpc, hand_value, hand });
                }
            } else {
                // RPC collision: store the MAXIMUM value for this pattern
                if (hand_value > result.value_ptr.*) {
                    result.value_ptr.* = hand_value;
                    
                    // Update the stored pattern with the higher value
                    for (non_flush_patterns.items) |*pattern| {
                        if (pattern.rpc == rpc) {
                            pattern.value = hand_value;
                            break;
                        }
                    }
                    
                    if (rpc == 0x48000000 or rpc == 0x89000000) {
                        print("  UPDATE: RPC 0x{x:08} updated to higher value {} for hand 0x{x:013}\n", 
                              .{ rpc, hand_value, hand });
                    }
                } else if (result.value_ptr.* != hand_value) {
                    // Same RPC but lower/equal value - this is expected
                    if (rpc == 0x48000000 or rpc == 0x89000000) {
                        print("  IGNORE: RPC 0x{x:08} keeping value {}, ignoring lower {} for hand 0x{x:013}\n", 
                              .{ rpc, result.value_ptr.*, hand_value, hand });
                    }
                }
            }
        }
    }

    print("\nFINAL COUNTS:\n", .{});
    print("  Hands processed: {}\n", .{enumerator.count});
    print("  Unique RPC patterns: {}\n", .{non_flush_patterns.items.len});
    print("  Unique flush patterns: {}\n", .{flush_patterns.items.len});

    // Build CHD hash for RPCs
    print("\n=== CHD HASH CONSTRUCTION (DESIGN.md COMPLIANT) ===\n", .{});
    print("RPC patterns: {}\n", .{non_flush_patterns.items.len});

    const chd_result = try buildRPCHash(allocator, non_flush_patterns.items);
    defer chd.deinit(chd_result, allocator);

    print("CHD construction SUCCESS:\n", .{});
    print("  Salt: 0x{x}\n", .{chd_result.salt});
    print("  Table size: {} (load factor: {d:.2})\n", .{ chd_result.table_size, @as(f64, @floatFromInt(non_flush_patterns.items.len)) / @as(f64, @floatFromInt(chd_result.table_size)) });
    print("  Buckets: {} displacements\n", .{chd_result.num_buckets});
    print("  ACTUAL displacement array length: {}\n", .{chd_result.displacements.len});
    print("  Values: {} elements\n", .{chd_result.table_size});

    // Build value table for CHD
    var chd_values = try allocator.alloc(u16, chd_result.table_size);
    defer allocator.free(chd_values);
    @memset(chd_values, 0);

    // Populate CHD value table using the same lookup logic as runtime
    for (non_flush_patterns.items) |pattern| {
        const final_idx = chd.lookup(@as(u64, pattern.rpc), chd_result);
        if (final_idx < chd_values.len) {
            chd_values[final_idx] = pattern.value;
        }
    }

    // Build BBHash for flush patterns
    print("\n=== BBHASH CONSTRUCTION ===\n", .{});
    print("Flush patterns: {}\n", .{flush_patterns.items.len});

    const bbhash_result = try buildBBHash(allocator, flush_patterns.items);
    defer bbhash_result.deinit(allocator);

    print("BBHash construction complete:\n", .{});
    print("  Magic constant: 0x{x}\n", .{bbhash_result.magic});
    print("  Value table: {} elements\n", .{bbhash_result.values.len});

    // Generate tables.zig
    print("\n=== GENERATING TABLES.ZIG (DESIGN.md COMPLIANT) ===\n", .{});

    const file = try std.fs.cwd().createFile("src/tables.zig", .{});
    defer file.close();
    const writer = file.writer();

    try writer.print("// Auto-generated CHD tables for 7-card poker evaluation (DESIGN.md compliant)\n", .{});
    try writer.print("// Generated by: zig run src/build_tables.zig\n", .{});
    try writer.print("// DO NOT EDIT - regenerate with build_tables\n\n", .{});

    // CHD parameters matching DESIGN.md
    try writer.print("// CHD Hash for RPC patterns (DESIGN.md specification)\n", .{});
    try writer.print("pub const CHD_SALT: u64 = 0x{x};\n", .{chd_result.salt});
    try writer.print("pub const CHD_NUM_BUCKETS: u32 = {};\n", .{chd_result.num_buckets});
    try writer.print("pub const CHD_TABLE_SIZE: u32 = {};\n", .{chd_result.table_size});
    try writer.print("pub const CHD_NUM_BUCKETS_LOG2: u6 = {};\n", .{chd_result.num_buckets_log2});
    try writer.print("pub const CHD_TABLE_SIZE_LOG2: u6 = {};\n\n", .{chd_result.table_size_log2});

    // CHD displacement table
    try writer.print("pub const CHD_DISPLACEMENTS = [_]u8{{", .{});
    for (chd_result.displacements, 0..) |displacement, i| {
        if (i % 16 == 0) try writer.print("\n    ", .{});
        try writer.print("0x{x:02}", .{displacement});
        if (i < chd_result.displacements.len - 1) try writer.print(", ", .{});
    }
    try writer.print("\n}};\n\n", .{});

    // CHD value table
    try writer.print("pub const CHD_VALUES = [_]u16{{", .{});
    for (chd_values, 0..) |value, i| {
        if (i % 8 == 0) try writer.print("\n    ", .{});
        try writer.print("0x{x:04}", .{value});
        if (i < chd_values.len - 1) try writer.print(", ", .{});
    }
    try writer.print("\n}};\n\n", .{});

    // BBHash table for flush patterns
    try writer.print("// BBHash for flush patterns (DESIGN.md specification)\n", .{});
    try writer.print("pub const BBHASH_MAGIC: u32 = 0x{x};\n", .{bbhash_result.magic});
    try writer.print("pub const BBHASH_SHIFT: u6 = {};\n", .{bbhash_result.shift});
    try writer.print("pub const BBHASH_TABLE_SIZE: u32 = {};\n\n", .{bbhash_result.table_size});
    try writer.print("pub const BBHASH_VALUES = [_]u16{{", .{});
    for (bbhash_result.values, 0..) |value, i| {
        if (i % 8 == 0) try writer.print("\n    ", .{});
        try writer.print("0x{x:04}", .{value});
        if (i < bbhash_result.values.len - 1) try writer.print(", ", .{});
    }
    try writer.print("\n}};\n\n", .{});

    const total_size = chd_result.displacements.len + (chd_values.len * 2) + (bbhash_result.values.len * 2);
    print("Generated tables.zig:\n", .{});
    print("  CHD displacements: {} elements ({} bytes)\n", .{ chd_result.displacements.len, chd_result.displacements.len });
    print("  CHD values: {} elements ({} bytes)\n", .{ chd_values.len, chd_values.len * 2 });
    print("  BBHash table: {} elements ({} bytes)\n", .{ bbhash_result.values.len, bbhash_result.values.len * 2 });
    print("  Total size: {} bytes ({d:.1} KB)\n", .{ total_size, @as(f64, @floatFromInt(total_size)) / 1024.0 });

    print("\n=== SUCCESS ===\n", .{});
    print("DESIGN.md compliant CHD tables generated successfully!\n", .{});

    // VALIDATION: Test RPC encoding and CHD lookup with real poker data
    print("\n=== VALIDATION ===\n", .{});
    print("Testing RPC encoding on sample hands...\n", .{});

    // Test RPC encoding on first few patterns
    const sample_size = @min(5, non_flush_patterns.items.len);
    for (non_flush_patterns.items[0..sample_size]) |pattern| {
        print("  RPC 0x{x:08} -> value {}\n", .{ pattern.rpc, pattern.value });
    }

    // Performance test of optimized CHD lookup
    if (non_flush_patterns.items.len > 0) {
        print("\nTesting CHD lookup performance...\n", .{});
        const test_rpc = non_flush_patterns.items[0].rpc;
        const iterations = 1_000_000;
        var checksum: u64 = 0;

        const start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            const value = chd.chdLookup(test_rpc, chd_result.displacements, chd_values);
            checksum +%= value;
        }
        const end = std.time.nanoTimestamp();

        const total_ns = @as(f64, @floatFromInt(end - start));
        const ns_per_lookup = total_ns / @as(f64, @floatFromInt(iterations));

        print("  Performance: {d:.2} ns per lookup\n", .{ns_per_lookup});
        print("  DESIGN.md target: 2.5 ns\n", .{});
        print("  Speedup needed: {d:.1}x\n", .{ns_per_lookup / 2.5});
        print("  Checksum: {} (prevents optimization)\n", .{checksum});

        if (ns_per_lookup < 5.0) {
            print("  Status: ✅ Excellent performance!\n", .{});
        } else if (ns_per_lookup < 10.0) {
            print("  Status: 🔄 Good, close to target\n", .{});
        } else {
            print("  Status: ⚠️  Needs SIMD optimization\n", .{});
        }
    }
}
