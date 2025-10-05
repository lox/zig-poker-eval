// Minimal Perfect Hash Function (CHD) implementation
// Pure primitives - no dependencies on specific table implementations

const std = @import("std");

// Default CHD configuration
pub const DEFAULT_NUM_BUCKETS = 8192; // 2^13 buckets
pub const DEFAULT_TABLE_SIZE = 131072; // 2^17 slots
pub const DEFAULT_MAGIC_CONSTANT = 0x9e3779b97f4a7c15;

// Hash function used for CHD
inline fn mix64(x: u64, magic_constant: u64) u64 {
    const h = x *% magic_constant;
    return h ^ (h >> 29);
}

// CHD hash result type
pub const HashResult = struct {
    bucket: u32,
    base_index: u32,
};

// Pattern type for CHD construction
pub const Pattern = struct {
    key: u32,
    value: u16,
};

// CHD construction result
pub const CHDResult = struct {
    g_array: []u8,
    value_table: []u16,
    magic_constant: u64,
};

// CHD lookup function (primitive)
pub inline fn lookup(key: u32, magic_constant: u64, g_array: []const u8, value_table: []const u16, table_size: u32) u16 {
    const h = mix64(@as(u64, key), magic_constant);
    const bucket = @as(u32, @intCast(h >> 51)); // Top 13 bits
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits
    const displacement = g_array[bucket];
    const final_index = (base_index + displacement) & (table_size - 1);
    return value_table[final_index];
}

// CHD hash function for table construction
pub fn hashKey(key: u32, magic_constant: u64) HashResult {
    const h = mix64(@as(u64, key), magic_constant);
    return .{
        .bucket = @intCast(h >> 51),
        .base_index = @intCast(h & 0x1FFFF),
    };
}

// Build CHD perfect hash table
pub fn buildChd(allocator: std.mem.Allocator, patterns: []const Pattern, num_buckets: u32, table_size: u32) !CHDResult {
    var g_array = try allocator.alloc(u8, num_buckets);
    var value_table = try allocator.alloc(u16, table_size);

    // Try different seeds until one works
    for (0..10) |attempt| {
        const magic_constant = DEFAULT_MAGIC_CONSTANT +% (attempt * 0x123456789abcdef);

        // Reset tables
        @memset(g_array, 0);
        @memset(value_table, 0);

        // Group patterns by bucket
        var buckets = try allocator.alloc(std.ArrayList(Pattern), num_buckets);
        for (buckets) |*bucket| {
            bucket.* = .empty;
        }
        defer {
            for (buckets) |*bucket| bucket.deinit(allocator);
            allocator.free(buckets);
        }

        for (patterns) |pattern| {
            const h = hashKey(pattern.key, magic_constant);

            // Debug tracking for problem RPCs
            if (pattern.key == 742203275 or pattern.key == 60953155) {
                std.debug.print("CHD bucket assignment: RPC {} -> bucket {} (base_index {})\n", .{ pattern.key, h.bucket, h.base_index });
            }

            try buckets[h.bucket].append(allocator, pattern);
        }

        // Try to find displacement for each bucket
        var occupied = try allocator.alloc(bool, table_size);
        defer allocator.free(occupied);
        @memset(occupied, false);

        var success = true;

        // Process buckets in order of decreasing size
        var bucket_order = try allocator.alloc(u32, num_buckets);
        defer allocator.free(bucket_order);
        for (0..num_buckets) |i| {
            bucket_order[i] = @intCast(i);
        }
        std.sort.pdq(u32, bucket_order, buckets, bucketSizeDesc);

        for (bucket_order) |bucket_id| {
            const bucket = &buckets[bucket_id];
            if (bucket.items.len == 0) continue;

            const displacement = findDisplacement(bucket.items, occupied, magic_constant, table_size) orelse {
                success = false;
                break;
            };

            g_array[bucket_id] = @intCast(displacement);

            // Place all entries
            for (bucket.items) |pattern| {
                const h = hashKey(pattern.key, magic_constant);
                const slot = (h.base_index + displacement) & (table_size - 1);

                // Debug tracking for problem RPC
                if (pattern.key == 742203275) {
                    std.debug.print("CHD: Placing RPC {} (value {}) at slot {} (bucket {}, base_index {}, displacement {})\n", .{ pattern.key, pattern.value, slot, h.bucket, h.base_index, displacement });
                }
                if (slot == 93133) {
                    std.debug.print("CHD: Writing value {} to slot 93133 (RPC {})\n", .{ pattern.value, pattern.key });
                }

                occupied[slot] = true;
                value_table[slot] = pattern.value;
            }
        }

        if (success) {
            return CHDResult{
                .g_array = g_array,
                .value_table = value_table,
                .magic_constant = magic_constant,
            };
        }
    }

    allocator.free(g_array);
    allocator.free(value_table);
    return error.CHDConstructionFailed;
}

fn findDisplacement(patterns: []const Pattern, occupied: []bool, magic_constant: u64, table_size: u32) ?u32 {
    for (0..256) |d| {
        var collision = false;
        var slots_to_check = std.AutoHashMap(u32, void).init(std.heap.page_allocator);
        defer slots_to_check.deinit();

        for (patterns) |pattern| {
            const h = hashKey(pattern.key, magic_constant);
            const slot = (h.base_index + @as(u32, @intCast(d))) & (table_size - 1);

            // Check if slot is already occupied OR if we've already seen this slot in this bucket
            if (occupied[slot] or slots_to_check.contains(slot)) {
                collision = true;

                // Debug for our problem case
                if (pattern.key == 742203275 or pattern.key == 60953155) {
                    std.debug.print("CHD findDisplacement: RPC {} collides at slot {} with displacement {}\n", .{ pattern.key, slot, d });
                }
                break;
            }
            slots_to_check.put(slot, {}) catch {};
        }
        if (!collision) return @intCast(d);
    }
    return null;
}

fn bucketSizeDesc(buckets: []std.ArrayList(Pattern), a: u32, b: u32) bool {
    return buckets[a].items.len > buckets[b].items.len;
}
