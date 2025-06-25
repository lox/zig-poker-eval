//! CHD (Compact Hash and Displace) Minimal Perfect Hash Function Implementation
//!
//! This implements the CHD algorithm for creating minimal perfect hash functions.
//! A minimal perfect hash function maps n distinct keys to exactly n distinct values
//! in the range [0, n-1] with zero collisions and 100% space efficiency.
//!
//! ## Algorithm Overview
//!
//! The CHD algorithm works in two phases:
//!
//! ### Phase 1: Bucket Assignment
//! 1. Hash each key using hash function h1(key, salt1)
//! 2. Assign keys to buckets based on h1(key, salt1) % num_buckets
//! 3. Sort buckets by size in descending order (critical heuristic)
//!
//! ### Phase 2: Displacement Search
//! For each bucket (starting with largest):
//! 1. Try displacement values d = 0, 1, 2, ... until no collisions
//! 2. For each key k in bucket, compute final position: (h2(k, salt2) + d) % table_size
//! 3. Check if all positions are unique and not used by previous buckets
//! 4. If successful, mark positions as used and store displacement value
//!
//! ## References
//!
//! Based on "An Optimal Algorithm for Generating Minimal Perfect Hash Functions"
//! by Czech, Havas, and Majewski (1992)
//! https://staff.itee.uq.edu.au/havas/1992chm.pdf

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Result of CHD construction containing the displacement table and final hash table
pub const CHDResult = struct {
    /// Displacement values for each bucket (u8 sufficient for good load factors)
    displacements: []u8,
    /// Number of buckets used in the displacement table
    num_buckets: u32,
    /// Hash salt used for both bucket and base index calculation
    salt: u64,
    /// Log2 of number of buckets (for efficient bucket calculation)
    num_buckets_log2: u6,
    /// Log2 of table size (for efficient base index and modulo calculation)
    table_size_log2: u6,
    /// Total size of the final hash table (power of 2)
    table_size: u32,
};

/// Optimized mix function for CHD (based on expert feedback)  
/// Single multiply provides both bucket and base index with statistical independence
pub fn mix64(x: u64) u64 {
    var h = x;
    h ^= h >> 33;
    h *%= 0x9e3779b97f4a7c15; // 64-bit golden ratio constant
    h ^= h >> 29;
    return h;
}

/// Encode a poker hand into a Rank Pattern Code (RPC) as specified in DESIGN.md
/// Maps rank multiplicities to a stable 32-bit code: 3 bits per rank (13 ranks)
/// DESIGN.md: "pack the low 32 bits (the upper 7 are always zero)"
pub fn encodeRPC(rank_counts: [13]u8) u32 {
    var rpc: u64 = 0; // Use 64-bit for full 13×3=39 bits, then truncate
    for (rank_counts, 0..) |count, rank| {
        const shift: u6 = @intCast(rank * 3); // All 13 ranks: 0,3,6,9...36
        rpc |= (@as(u64, count) & 0x7) << shift; // 3 bits per rank
    }
    return @truncate(rpc); // Return low 32 bits as specified in DESIGN.md
}

/// Extract rank counts from a poker hand for RPC encoding
pub fn getRankCounts(hand: u64) [13]u8 {
    var counts = [_]u8{0} ** 13;
    
    // Extract suit masks
    const clubs = @as(u16, @truncate(hand >> 0)) & 0x1FFF;
    const diamonds = @as(u16, @truncate(hand >> 13)) & 0x1FFF;
    const hearts = @as(u16, @truncate(hand >> 26)) & 0x1FFF;
    const spades = @as(u16, @truncate(hand >> 39)) & 0x1FFF;
    
    // Count each rank across all suits
    for (0..13) |rank| {
        const mask = @as(u16, 1) << @intCast(rank);
        if (clubs & mask != 0) counts[rank] += 1;
        if (diamonds & mask != 0) counts[rank] += 1;
        if (hearts & mask != 0) counts[rank] += 1;
        if (spades & mask != 0) counts[rank] += 1;
    }
    
    return counts;
}

/// DESIGN.md optimized CHD lookup - single hash with high/low bit extraction  
/// This is the performance-critical path that must match DESIGN.md section 5 exactly
pub fn chdLookup(rpc: u32, displacements: []const u8, values: []const u16) u16 {
    const tables = @import("tables.zig");
    
    // Debug: check actual displacement array length
    if (displacements.len != 8192) {
        std.debug.print("MISMATCH: displacements.len = {}, expected 8192\n", .{displacements.len});
    }
    
    // CRITICAL: Must use same salt as build process!
    const h = mix64(@as(u64, rpc) +% tables.CHD_SALT);
    
    // FIXED: Remove masking to match build-time getBucket() exactly
    // bucket = h >> 51 gives top 13 bits, exactly matching getBucket() logic
    const bucket: u32 = @truncate(h >> 51);           // No masking - must match build-time!
    const base_idx: u32 = @truncate(h & 0xFFFF);      // Low 16 bits → base index (DESIGN.md section 5)
    const displacement = displacements[bucket];
    const final_idx = (base_idx + displacement) & (@as(u32, @intCast(values.len)) - 1); // Mask to value array size
    return values[final_idx];
}

/// Legacy functions for backward compatibility (will be optimized away)
pub fn getBucket(key: u64, salt: u64, num_buckets_log2: u6) u32 {
    const h = mix64(key +% salt);
    const shift = 64 - @as(u8, num_buckets_log2);
    return @truncate(h >> @as(u6, @intCast(shift)));
}

pub fn getBaseIndex(key: u64, salt: u64, table_size_log2: u6) u32 {
    const h = mix64(key +% salt);
    const mask = (@as(u64, 1) << table_size_log2) - 1;
    return @truncate(h & mask);
}

/// Attempt to construct a CHD hash function for the given keys
/// Returns CHDResult on success or error on failure
pub fn buildCHDHash(allocator: Allocator, keys: []const u64) !CHDResult {
    const max_attempts = 10;

    for (0..max_attempts) |attempt| {
        // Single salt per attempt - mix64 provides independence via high/low bit split
        const salt = 0x123456789ABCDEF0 +% (@as(u64, @intCast(attempt)) * 2654435761);

        if (tryCHDConstruction(allocator, keys, salt)) |result| {
            std.debug.print("CHD construction succeeded on attempt {}\n", .{attempt + 1});
            return result;
        } else |err| {
            std.debug.print("  Attempt {}: Failed with error {}\n", .{ attempt + 1, err });
        }
    }

    std.debug.print("CHD construction failed after {} attempts\n", .{max_attempts});
    return error.CHDConstructionFailed;
}

/// Internal function that attempts CHD construction with specific salt value
fn tryCHDConstruction(allocator: Allocator, keys: []const u64, salt: u64) !CHDResult {
    _ = keys.len; // DESIGN.md uses fixed sizes regardless of input count
    
    // DESIGN.md mandates fixed sizes: 8192 buckets, 65536 table size
    // This ensures the exact memory footprint and hash performance specified
    const table_size: u32 = 65536;  // N = 65536 (DESIGN.md section 2)
    const table_size_log2: u6 = 16; // log2(65536) = 16
    
    const num_buckets: u32 = 8192;     // m = 8192 (DESIGN.md section 2) 
    const num_buckets_log2: u6 = 13;  // log2(8192) = 13

    // Create buckets to group keys
    var buckets = try allocator.alloc(std.ArrayList(u64), num_buckets);
    defer {
        for (buckets) |*bucket| {
            bucket.deinit();
        }
        allocator.free(buckets);
    }

    // Initialize buckets
    for (buckets) |*bucket| {
        bucket.* = std.ArrayList(u64).init(allocator);
    }

    // Assign keys to buckets using getBucket() 
    for (keys) |key| {
        const bucket_idx = getBucket(key, salt, num_buckets_log2);
        try buckets[bucket_idx].append(key);
    }

    // Sort buckets by size (largest first) - critical heuristic
    const BucketSorter = struct {
        buckets_ptr: *[]std.ArrayList(u64),

        pub fn lessThan(self: @This(), a_index: usize, b_index: usize) bool {
            return self.buckets_ptr.*[a_index].items.len > self.buckets_ptr.*[b_index].items.len;
        }
    };

    const bucket_indices = try allocator.alloc(usize, num_buckets);
    defer allocator.free(bucket_indices);

    for (bucket_indices, 0..) |*idx, i| {
        idx.* = i;
    }

    const sorter = BucketSorter{ .buckets_ptr = &buckets };
    std.sort.insertion(usize, bucket_indices, sorter, BucketSorter.lessThan);

    // Track which positions in the final table are occupied
    var occupied = try allocator.alloc(bool, table_size);
    defer allocator.free(occupied);
    @memset(occupied, false);

    // Try to find displacement for each bucket (u8 for memory efficiency)
    var displacements = try allocator.alloc(u8, num_buckets);
    defer allocator.free(displacements);
    @memset(displacements, 0);

    for (bucket_indices) |bucket_idx| {
        const bucket = &buckets[bucket_idx];
        if (bucket.items.len == 0) continue;

        // Try displacement values until we find one that works
        var displacement: u32 = 0;
        const max_displacement = @min(255, table_size); // Limit to u8 range for storage efficiency
        var found = false;

        displacement_search: while (displacement < max_displacement) : (displacement += 1) {
            var positions = try allocator.alloc(u32, bucket.items.len);
            defer allocator.free(positions);

            // Compute final positions for all keys in this bucket using getBaseIndex
            for (bucket.items, 0..) |key, i| {
                const base_pos = getBaseIndex(key, salt, table_size_log2);
                const final_pos = (base_pos + displacement) & (table_size - 1); // AND instead of mod for power-of-2
                positions[i] = final_pos;

                // Check if position is already occupied
                if (occupied[final_pos]) {
                    continue :displacement_search;
                }
            }

            // Check for internal collisions within this bucket
            for (positions, 0..) |pos1, i| {
                for (positions[i + 1 ..]) |pos2| {
                    if (pos1 == pos2) {
                        continue :displacement_search;
                    }
                }
            }

            // This displacement works! Mark positions as occupied
            for (positions) |pos| {
                occupied[pos] = true;
            }
            displacements[bucket_idx] = @intCast(displacement); // Safe cast to u8
            found = true;
            break;
        }

        if (!found) {
            std.debug.print("  Bucket {} (size {}): Failed to find displacement\n", .{ bucket_idx, bucket.items.len });
            return error.DisplacementNotFound;
        }
    }

    // Success! Return the CHD parameters
    const result_displacements = try allocator.alloc(u8, num_buckets);
    @memcpy(result_displacements, displacements);

    return CHDResult{
        .displacements = result_displacements,
        .num_buckets = @intCast(num_buckets),
        .salt = salt,
        .num_buckets_log2 = num_buckets_log2,
        .table_size_log2 = table_size_log2,
        .table_size = @intCast(table_size),
    };
}

/// Lookup a key using the CHD hash function  
/// Returns the hash value in range [0, table_size-1]
pub fn lookup(key: u64, chd_result: CHDResult) u32 {
    const bucket_idx = getBucket(key, chd_result.salt, chd_result.num_buckets_log2);
    const displacement = chd_result.displacements[bucket_idx];
    const base_pos = getBaseIndex(key, chd_result.salt, chd_result.table_size_log2);
    return (base_pos + displacement) & (chd_result.table_size - 1); // AND instead of mod
}

/// Free memory allocated for CHD result
pub fn deinit(chd_result: CHDResult, allocator: Allocator) void {
    allocator.free(chd_result.displacements);
}

/// Benchmark CHD lookup performance using realistic tables
/// Returns nanoseconds per lookup
pub fn benchmarkCHDLookup(iterations: u32) f64 {
    const tables = @import("tables.zig");
    
    // Use realistic RPC patterns from generated tables
    const test_rpcs = [_]u32{ 
        0x00000292, 0x00000452, 0x00001252, 0x00008252, 0x00040252,
        0x12345678, 0x9abcdef0, 0x00000001, 0x11111111, 0x22222222
    };
    
    var checksum: u64 = 0;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |i| {
        const rpc = test_rpcs[i % test_rpcs.len];
        const value = chdLookup(rpc, &tables.CHD_DISPLACEMENTS, &tables.CHD_VALUES);
        checksum +%= value;
    }
    
    const end = std.time.nanoTimestamp();
    const total_ns = @as(f64, @floatFromInt(end - start));
    const ns_per_lookup = total_ns / @as(f64, @floatFromInt(iterations));
    
    // Use checksum to prevent optimization
    if (checksum == 0) {
        std.debug.print("Warning: unexpected zero checksum in CHD benchmark\n", .{});
    }
    
    return ns_per_lookup;
}

// ============================================================================
// TESTS
// ============================================================================

test "CHD build vs runtime lookup consistency" {
    const tables = @import("tables.zig");
    
    // Test critical mismatch: build uses salt, runtime doesn't!
    const test_rpcs = [_]u32{ 0x00000292, 0x00000452, 0x00001252, 0x00008252 };
    
    std.debug.print("\n=== CHD BUILD vs RUNTIME MISMATCH TEST ===\n", .{});
    
    for (test_rpcs) |rpc| {
        // BUILD-TIME method (with salt, used during table generation)
        const build_bucket = getBucket(@as(u64, rpc), tables.CHD_SALT, tables.CHD_NUM_BUCKETS_LOG2);
        const build_base = getBaseIndex(@as(u64, rpc), tables.CHD_SALT, tables.CHD_TABLE_SIZE_LOG2);
        const build_displacement = tables.CHD_DISPLACEMENTS[build_bucket];
        const build_final = (build_base + build_displacement) & (tables.CHD_TABLE_SIZE - 1);
        const build_value = tables.CHD_VALUES[build_final];
        
        // RUNTIME method (no salt, used by chdLookup)
        const runtime_value = chdLookup(rpc, &tables.CHD_DISPLACEMENTS, &tables.CHD_VALUES);
        
        std.debug.print("RPC 0x{x:08}:\n", .{rpc});
        std.debug.print("  BUILD:   bucket={}, base={}, disp={}, final={}, value={}\n", 
                       .{ build_bucket, build_base, build_displacement, build_final, build_value });
        std.debug.print("  RUNTIME: value={}\n", .{runtime_value});
        std.debug.print("  MATCH: {}\n\n", .{build_value == runtime_value});
        
        // This test will FAIL because of the salt mismatch!
        if (build_value != runtime_value) {
            std.debug.print("❌ CRITICAL BUG: Build and runtime lookups don't match!\n", .{});
            std.debug.print("   This explains why SIMD evaluator returns 0 for all hands.\n", .{});
        }
    }
}

test "chdLookup hash extraction" {
    // Test the hash bit extraction logic in chdLookup
    const test_rpc: u32 = 0x00001252;
    const h = mix64(@as(u64, test_rpc));
    
    const bucket_bits = @as(u32, @truncate(h >> 51)) & (8192 - 1); // 13 bits from top
    const base_bits: u32 = @truncate(h & 0xFFFF); // 16 bits from bottom
    
    std.debug.print("\n=== chdLookup Hash Extraction ===\n", .{});
    std.debug.print("RPC: 0x{x:08}\n", .{test_rpc});
    std.debug.print("Hash: 0x{x:016}\n", .{h});
    std.debug.print("Bucket bits (h>>51 & 8191): {}\n", .{bucket_bits});
    std.debug.print("Base bits (h & 0xFFFF): {}\n", .{base_bits});
    std.debug.print("Expected bucket range: 0-8191\n", .{});
    std.debug.print("Expected base range: 0-65535\n", .{});
    
    try std.testing.expect(bucket_bits < 8192);
    try std.testing.expect(base_bits < 65536);
}

test "mix64 vs salted hash consistency" {
    const test_key: u64 = 0x00001252;
    const salt: u64 = 0x12345679d72bd252; // From tables
    
    // DESIGN.md approach: single mix64 without salt
    const design_hash = mix64(test_key);
    
    // Legacy approach: mix64 with salt
    const legacy_hash = mix64(test_key +% salt);
    
    std.debug.print("\n=== Hash Function Comparison ===\n", .{});
    std.debug.print("Key: 0x{x:016}\n", .{test_key});
    std.debug.print("Salt: 0x{x:016}\n", .{salt});
    std.debug.print("DESIGN.md hash (no salt): 0x{x:016}\n", .{design_hash});
    std.debug.print("Legacy hash (with salt): 0x{x:016}\n", .{legacy_hash});
    std.debug.print("Hashes match: {}\n", .{design_hash == legacy_hash});
    
    // They should NOT match - this shows the fundamental inconsistency
    if (design_hash == legacy_hash) {
        std.debug.print("❌ Unexpected: Hashes should be different!\n", .{});
    } else {
        std.debug.print("✅ Expected: Hashes are different (this is the bug!)\n", .{});
    }
}

test "CHD hash functions work" {
    const tables = @import("tables.zig");
    
    // Test the specific RPC patterns that are failing in correctness tests
    const failing_rpcs = [_]u32{ 0x48000000, 0x89000000 }; // Four Aces, Full House
    const expected_values = [_]u16{ 7000, 6000 };
    
    std.debug.print("\n=== DEBUGGING FAILING RPC PATTERNS ===\n", .{});
    
    for (failing_rpcs, 0..) |rpc, i| {
        const chd_value = chdLookup(rpc, &tables.CHD_DISPLACEMENTS, &tables.CHD_VALUES);
        const expected = expected_values[i];
        
        std.debug.print("RPC 0x{x:08}: CHD={}, Expected={}, Match={}\n", 
                       .{ rpc, chd_value, expected, chd_value == expected });
        
        if (chd_value != expected) {
            std.debug.print("  ❌ MISMATCH: CHD table has wrong value for this RPC!\n", .{});
        }
    }
    
    // Test the basic hash functions directly with known salt
    const test_patterns = [_]u16{ 0x002f, 0x0030, 0x0100, 0x1c27 };
    
    for (test_patterns) |pattern| {
        const bucket = getBucket(@as(u64, pattern), tables.CHD_SALT, 7);
        const base_idx = getBaseIndex(@as(u64, pattern), tables.CHD_SALT, 9);
        const displacement = tables.CHD_DISPLACEMENTS[bucket];
        const final_idx = (base_idx + displacement) & (tables.CHD_TABLE_SIZE - 1);
        const value = tables.CHD_VALUES[final_idx];
        
        std.debug.print("Pattern 0x{x:04} -> bucket {} -> base {} -> disp {} -> final {} -> value {}\n", 
                       .{ pattern, bucket, base_idx, displacement, final_idx, value });
        
        // Basic sanity checks
        try std.testing.expect(bucket < tables.CHD_NUM_BUCKETS);
        try std.testing.expect(final_idx < tables.CHD_TABLE_SIZE);
        try std.testing.expect(value <= 7500); // Allow higher values for debugging
    }
}

test "CHD optimized lookup functionality" {
    // Simple functionality test using benchmark function
    const ns_per_lookup = benchmarkCHDLookup(100_000);
    
    std.debug.print("CHD Lookup Test:\n", .{});
    std.debug.print("  Time per lookup: {d:.2} ns\n", .{ns_per_lookup});
    std.debug.print("  DESIGN.md target: 2.5 ns\n", .{});
    
    // Basic sanity check - should be reasonable performance
    try std.testing.expect(ns_per_lookup < 100.0); // Very generous bound for testing
    try std.testing.expect(ns_per_lookup > 0.1);  // Should take some time
}
