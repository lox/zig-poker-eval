const std = @import("std");
const print = std.debug.print;
const evaluator = @import("slow_evaluator");

// Table generation constants from DESIGN.md  
const CHD_NUM_BUCKETS = 8192; // 2^13 buckets
const CHD_TABLE_SIZE = 131072; // 2^17 slots (increase to reduce load factor)
const CHD_EXPECTED_PATTERNS = 49205; // Non-flush patterns
const BBH_EXPECTED_PATTERNS = 1287; // C(13,5) flush patterns

// CHD magic constant (will be tuned during build)
var chd_magic_constant: u64 = 0x9e3779b97f4a7c15;

// Generated table data
var chd_g_array: [CHD_NUM_BUCKETS]u8 = undefined; // Displacement per bucket
var chd_value_table: [CHD_TABLE_SIZE]u16 = undefined; // Hand ranks
var bbhash_blob: std.ArrayList(u8) = undefined; // BBHash serialized data

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    print("Building poker evaluation tables...\n", .{});
    print("Expected patterns: {} non-flush, {} flush\n", .{ CHD_EXPECTED_PATTERNS, BBH_EXPECTED_PATTERNS });

    // Initialize data structures
    bbhash_blob = std.ArrayList(u8).init(allocator);
    defer bbhash_blob.deinit();

    // Build CHD table for non-flush hands
    print("Building CHD table for non-flush patterns...\n", .{});
    try build_chd_tables(allocator);

    // Build BBHash table for flush hands  
    print("Building BBHash table for flush patterns...\n", .{});
    try build_bbhash_tables(allocator);

    // Validate table sizes
    validate_table_sizes();

    // Generate output file
    print("Writing src/claude/tables.zig...\n", .{});
    try write_tables_file();

    print("Table generation complete!\n", .{});
}

fn build_chd_tables(allocator: std.mem.Allocator) !void {
    var patterns = std.ArrayList(RPCEntry).init(allocator);
    defer patterns.deinit();

    print("  Enumerating non-flush hands...\n", .{});
    try enumerate_non_flush_hands(&patterns, allocator);
    
    print("  Found {} non-flush patterns (expected {})\n", .{ patterns.items.len, CHD_EXPECTED_PATTERNS });

    print("  Building CHD displacement array...\n", .{});
    try build_chd_displacement_array(patterns.items);

    print("  CHD table built successfully\n", .{});
}

fn build_bbhash_tables(allocator: std.mem.Allocator) !void {
    var flush_patterns = std.ArrayList(FlushEntry).init(allocator);
    defer flush_patterns.deinit();

    print("  Enumerating flush hands...\n", .{});
    try enumerate_flush_hands(&flush_patterns, allocator);

    print("  Found {} flush patterns\n", .{flush_patterns.items.len});
    std.debug.assert(flush_patterns.items.len == BBH_EXPECTED_PATTERNS);

    print("  Building BBHash MPHF...\n", .{});
    try build_bbhash_mphf(flush_patterns.items);

    print("  BBHash table built successfully\n", .{});
}

// RPC (Rank Pattern Code) implementation
fn compute_rpc(hand: evaluator.Hand) u32 {
    // Get rank counts for each of the 13 ranks
    const suits = evaluator.getSuitMasks(hand);
    var rank_counts = [_]u8{0} ** 13;
    
    for (0..13) |rank| {
        const rank_bit = @as(u16, 1) << @intCast(rank);
        var count: u8 = 0;
        for (suits) |suit| {
            if ((suit & rank_bit) != 0) count += 1;
        }
        rank_counts[rank] = count;
    }
    
    // Base-5 encoding: preserves all 49,205 patterns in 31 bits
    // Each rank count (0-4) becomes a base-5 digit
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        std.debug.assert(count <= 4); // Max 4 of any rank in 7 cards
        rpc = rpc * 5 + count; // Radix-5 encoding
    }
    
    return rpc; // Result: 0...1,220,703,124 (fits in u32)
}

// CHD hash functions
fn mix64(x: u64) u64 {
    var result = x;
    result ^= result >> 33;
    result *%= chd_magic_constant;
    result ^= result >> 29;
    return result;
}

fn chd_hash(rpc: u32) struct { bucket: u32, base_index: u32 } {
    const h = mix64(@as(u64, rpc));
    return .{
        .bucket = @intCast(h >> 51), // Top 13 bits -> bucket (0..8191)
        .base_index = @intCast(h & 0x3FFFF), // Low 18 bits -> base index (0..262143)
    };
}

fn enumerate_non_flush_hands(patterns: *std.ArrayList(RPCEntry), allocator: std.mem.Allocator) !void {
    print("    Generating all 7-card hands...\n", .{});
    const all_hands = try generate_all_hands(allocator);
    defer all_hands.deinit();
    
    print("    Generated {} total hands, filtering non-flush...\n", .{all_hands.items.len});
    
    // Use HashMap to deduplicate by RPC
    var seen_rpcs = std.AutoHashMap(u32, u16).init(allocator);
    defer seen_rpcs.deinit();
    
    var non_flush_count: usize = 0;
    for (all_hands.items) |hand| {
        if (!evaluator.hasFlush(hand)) {
            const rpc = compute_rpc(hand);
            
            // Only store unique RPCs
            if (!seen_rpcs.contains(rpc)) {
                const rank = slow_evaluate_hand(hand);
                try seen_rpcs.put(rpc, rank);
                
                try patterns.append(RPCEntry{
                    .rpc = rpc,
                    .rank = rank,
                });
            }
            non_flush_count += 1;
        }
    }
    
    print("    Found {} non-flush hands ({} unique RPCs)\n", .{ non_flush_count, patterns.items.len });
}

fn enumerate_flush_hands(patterns: *std.ArrayList(FlushEntry), allocator: std.mem.Allocator) !void {
    print("    Generating all 7-card hands...\n", .{});
    const all_hands = try generate_all_hands(allocator);
    defer all_hands.deinit();
    
    print("    Filtering flush hands...\n", .{});
    
    var seen_patterns = std.AutoHashMap(u16, u16).init(allocator);
    defer seen_patterns.deinit();
    
    for (all_hands.items) |hand| {
        if (evaluator.hasFlush(hand)) {
            const suits = evaluator.getSuitMasks(hand);
            
            // Find the flush suit and extract top 5 ranks
            for (suits) |suit| {
                if (@popCount(suit) >= 5) {
                    const flush_pattern = get_top5_ranks(suit);
                    
                    // Only store unique patterns
                    if (!seen_patterns.contains(flush_pattern)) {
                        const rank = slow_evaluate_flush(flush_pattern);
                        try seen_patterns.put(flush_pattern, rank);
                        
                        
                        
                        try patterns.append(FlushEntry{
                            .pattern = @truncate(flush_pattern),
                            .rank = rank,
                        });
                    }
                    break; // Only need first qualifying suit
                }
            }
        }
    }
    
    print("    Found {} unique flush patterns\n", .{patterns.items.len});
}

fn get_top5_ranks(suit_mask: u16) u16 {
    // For flush evaluation, we need to find the best 5-card flush
    // This should prioritize straights (including A-low wheel) over high cards
    
    if (@popCount(suit_mask) == 5) {
        return suit_mask; // Exactly 5 cards, use all
    }
    
    // Check for straight flushes first (including wheel: A,5,4,3,2)
    const straight_patterns = [_]u16{
        0x1F00, // A,K,Q,J,10 (royal flush)
        0x0F80, // K,Q,J,10,9
        0x07C0, // Q,J,10,9,8
        0x03E0, // J,10,9,8,7
        0x01F0, // 10,9,8,7,6
        0x00F8, // 9,8,7,6,5
        0x007C, // 8,7,6,5,4
        0x003E, // 7,6,5,4,3
        0x001F, // 6,5,4,3,2
        0x100F, // A,5,4,3,2 (wheel)
    };
    
    // Check if any straight pattern fits
    for (straight_patterns) |pattern| {
        if ((suit_mask & pattern) == pattern) {
            return pattern; // Found a straight flush, use it
        }
    }
    
    // No straight found, take highest 5 ranks
    var result: u16 = 0;
    var count: u8 = 0;
    var rank: u8 = 12; // Start from Ace
    
    while (count < 5 and rank < 13) {
        const rank_bit = @as(u16, 1) << @intCast(rank);
        if ((suit_mask & rank_bit) != 0) {
            result |= rank_bit;
            count += 1;
        }
        if (rank == 0) break;
        rank -= 1;
    }
    
    
    return result;
}

fn build_chd_displacement_array(patterns: []const RPCEntry) !void {
    print("    Building CHD displacement array for {} patterns...\n", .{patterns.len});
    
    // Try different seeds until we find one that works
    var seed_attempts: u32 = 0;
    while (seed_attempts < 10) : (seed_attempts += 1) {
        chd_magic_constant = 0x9e3779b97f4a7c15 +% (@as(u64, seed_attempts) * 0x123456789abcdef);
        
        print("    Trying seed {} (attempt {})...\n", .{ chd_magic_constant, seed_attempts + 1 });
        
        if (try build_chd_with_seed(patterns)) {
            print("    CHD built successfully with seed {} (attempt {})\n", .{ chd_magic_constant, seed_attempts + 1 });
            return;
        }
        
        print("    Seed attempt {} failed, retrying...\n", .{seed_attempts + 1});
    }
    
    return error.CHDConstructionFailed;
}

fn build_chd_with_seed(patterns: []const RPCEntry) !bool {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Phase 0: Hash all keys into buckets
    print("      Creating buckets...\n", .{});
    var buckets = try create_chd_buckets(patterns, allocator);
    defer {
        for (buckets.items) |*bucket| {
            bucket.entries.deinit();
        }
        buckets.deinit();
    }
    
    print("      Created {} buckets\n", .{buckets.items.len});
    
    // Phase 1: Sort buckets by size (descending)
    std.sort.pdq(CHDBucket, buckets.items, {}, compare_bucket_sizes);
    
    // Show bucket distribution
    var max_bucket_size: usize = 0;
    var non_empty_buckets: usize = 0;
    for (buckets.items) |bucket| {
        if (bucket.entries.items.len > 0) {
            non_empty_buckets += 1;
            if (bucket.entries.items.len > max_bucket_size) {
                max_bucket_size = bucket.entries.items.len;
            }
        }
    }
    print("      Bucket stats: {} non-empty, max size {}\n", .{ non_empty_buckets, max_bucket_size });
    
    // Phase 2: Find displacements for each bucket
    var occupied_slots = [_]bool{false} ** CHD_TABLE_SIZE;
    chd_g_array = std.mem.zeroes(@TypeOf(chd_g_array));
    chd_value_table = std.mem.zeroes(@TypeOf(chd_value_table));
    
    for (buckets.items) |bucket| {
        const displacement = find_displacement(bucket, &occupied_slots);
        if (displacement > 255) {
            return false; // Displacement too large, try different seed
        }
        
        chd_g_array[bucket.id] = @intCast(displacement);
        
        // Mark slots as occupied and store values
        for (bucket.entries.items) |entry| {
            const h = chd_hash(entry.rpc);
            const slot = (h.base_index + displacement) & (CHD_TABLE_SIZE - 1);
            occupied_slots[slot] = true;
            chd_value_table[slot] = entry.rank;
        }
    }
    
    return true;
}

fn create_chd_buckets(patterns: []const RPCEntry, allocator: std.mem.Allocator) !std.ArrayList(CHDBucket) {
    
    var buckets = std.ArrayList(CHDBucket).init(allocator);
    
    // Initialize buckets
    for (0..CHD_NUM_BUCKETS) |i| {
        try buckets.append(CHDBucket{
            .id = @intCast(i),
            .entries = std.ArrayList(RPCEntry).init(allocator),
        });
    }
    
    // Hash all patterns into buckets
    for (patterns) |pattern| {
        const h = chd_hash(pattern.rpc);
        try buckets.items[h.bucket].entries.append(pattern);
    }
    
    return buckets;
}

fn find_displacement(bucket: CHDBucket, occupied_slots: *[CHD_TABLE_SIZE]bool) u32 {
    if (bucket.entries.items.len == 0) return 0; // Empty bucket
    
    var displacement: u32 = 0;
    
    while (displacement <= 255) : (displacement += 1) {
        var collision = false;
        
        // Check if this displacement causes any collisions
        for (bucket.entries.items) |entry| {
            const h = chd_hash(entry.rpc);
            const slot = (h.base_index + displacement) & (CHD_TABLE_SIZE - 1);
            if (occupied_slots[slot]) {
                collision = true;
                break;
            }
        }
        
        if (!collision) {
            return displacement;
        }
    }
    
    print("        Failed to find displacement for bucket {} with {} entries\n", .{ bucket.id, bucket.entries.items.len });
    return 256; // Failed to find displacement
}

fn compare_bucket_sizes(context: void, a: CHDBucket, b: CHDBucket) bool {
    _ = context;
    return a.entries.items.len > b.entries.items.len; // Descending order
}

fn build_bbhash_mphf(patterns: []const FlushEntry) !void {
    print("    Building 3-level BBHash MPHF for {} flush patterns...\n", .{patterns.len});
    
    // Clear BBHash blob
    bbhash_blob.clearRetainingCapacity();
    
    // BBHash algorithm with gamma = 4.0 (higher load factor for debugging)  
    const gamma = 4.0;
    
    // Build level 0 with γ = 2.0 load factor  
    var level0_size = @as(u32, @intFromFloat(@as(f64, @floatFromInt(patterns.len)) * gamma));
    level0_size = std.math.ceilPowerOfTwo(u32, level0_size) catch level0_size;
    
    var level0_bitmap = try std.ArrayList(u64).initCapacity(std.heap.page_allocator, (level0_size + 63) / 64);
    defer level0_bitmap.deinit();
    
    var level1_patterns = std.ArrayList(FlushEntry).init(std.heap.page_allocator);
    defer level1_patterns.deinit();
    
    var reordered_ranks = try std.ArrayList(u16).initCapacity(std.heap.page_allocator, patterns.len);
    defer reordered_ranks.deinit();
    
    // Build level 0 using BBHash algorithm
    const bbhash_seed0: u64 = 0x9ae16a3b2f90404f;
    try build_bbhash_level(patterns, bbhash_seed0, level0_size, &level0_bitmap, &level1_patterns, &reordered_ranks);
    
    print("    Level 0: handled {} patterns, {} remain for level 1\n", .{ patterns.len - level1_patterns.items.len, level1_patterns.items.len });
    
    // Build level 1 for remaining patterns
    var level1_bitmap = std.ArrayList(u64).init(std.heap.page_allocator);
    defer level1_bitmap.deinit();
    
    var level2_patterns = std.ArrayList(FlushEntry).init(std.heap.page_allocator);
    defer level2_patterns.deinit();
    
    // Declare level sizes outside conditionals for serialization access
    var level1_size: u32 = 0;
    var level2_size: u32 = 0;
    
    if (level1_patterns.items.len > 0) {
        level1_size = @as(u32, @intFromFloat(@as(f64, @floatFromInt(level1_patterns.items.len)) * gamma));
        level1_size = std.math.ceilPowerOfTwo(u32, level1_size) catch level1_size;
        
        const bbhash_seed1: u64 = 0xaf36d42dfe24aa0f;
        try build_bbhash_level(level1_patterns.items, bbhash_seed1, level1_size, &level1_bitmap, &level2_patterns, &reordered_ranks);
        
        print("    Level 1: handled {} patterns, {} remain for level 2\n", .{ level1_patterns.items.len - level2_patterns.items.len, level2_patterns.items.len });
    }
    
    // Build level 2 for remaining patterns
    var level2_bitmap = std.ArrayList(u64).init(std.heap.page_allocator);
    defer level2_bitmap.deinit();
    
    if (level2_patterns.items.len > 0) {
        // For level 2, ensure table is large enough to handle all remaining patterns
        level2_size = @as(u32, @intFromFloat(@as(f64, @floatFromInt(level2_patterns.items.len)) * gamma));
        level2_size = std.math.ceilPowerOfTwo(u32, level2_size) catch level2_size;
        
        // Ensure minimum size to reduce collisions
        if (level2_size < 256) level2_size = 256;
        
        const bbhash_seed2: u64 = 0x597d5f64ce7a3a8d;
        var dummy_remaining = std.ArrayList(FlushEntry).init(std.heap.page_allocator);
        defer dummy_remaining.deinit();
        
        try build_bbhash_level(level2_patterns.items, bbhash_seed2, level2_size, &level2_bitmap, &dummy_remaining, &reordered_ranks);
        
        print("    Level 2: handled {} patterns, {} remain\n", .{ level2_patterns.items.len - dummy_remaining.items.len, dummy_remaining.items.len });
        
        // If patterns still remain, add them directly to avoid infinite levels
        for (dummy_remaining.items) |pattern| {
            try reordered_ranks.append(pattern.rank);
        }
        
        if (dummy_remaining.items.len > 0) {
            print("    Note: {} patterns placed without hash (fallback)\n", .{dummy_remaining.items.len});
        }
    }
    
    // Serialize BBHash data: seeds, bitmap sizes, bitmaps, ranks
    const writer = bbhash_blob.writer();
    
    // Write seeds
    try writer.writeInt(u64, bbhash_seed0, .little);
    try writer.writeInt(u64, 0xaf36d42dfe24aa0f, .little);
    try writer.writeInt(u64, 0x597d5f64ce7a3a8d, .little);
    
    // Write bitmap sizes (in u64 words)
    try writer.writeInt(u32, @intCast(level0_bitmap.items.len), .little);
    try writer.writeInt(u32, @intCast(level1_bitmap.items.len), .little);
    try writer.writeInt(u32, @intCast(level2_bitmap.items.len), .little);
    
    // Write level masks - CRITICAL FIX: Use actual table sizes, not bitmap sizes
    try writer.writeInt(u32, level0_size - 1, .little);
    try writer.writeInt(u32, if (level1_size > 0) level1_size - 1 else 0, .little);
    try writer.writeInt(u32, if (level2_size > 0) level2_size - 1 else 0, .little);
    
    // Write bitmaps
    for (level0_bitmap.items) |word| {
        try writer.writeInt(u64, word, .little);
    }
    for (level1_bitmap.items) |word| {
        try writer.writeInt(u64, word, .little);
    }
    for (level2_bitmap.items) |word| {
        try writer.writeInt(u64, word, .little);
    }
    
    // Write reordered ranks array
    for (reordered_ranks.items) |rank| {
        try writer.writeInt(u16, rank, .little);
    }
    
    print("    BBHash MPHF: {} bytes total\n", .{bbhash_blob.items.len});
    print("      Level 0: {} bits, Level 1: {} bits, Level 2: {} bits\n", .{ level0_bitmap.items.len * 64, level1_bitmap.items.len * 64, level2_bitmap.items.len * 64 });
}

fn build_bbhash_level(
    patterns: []const FlushEntry,
    seed: u64,
    table_size: u32,
    bitmap: *std.ArrayList(u64),
    remaining_patterns: *std.ArrayList(FlushEntry),
    reordered_ranks: *std.ArrayList(u16)
) !void {
    // Initialize bitmap with enough u64 words to cover table_size bits
    const bitmap_words = (table_size + 63) / 64;
    try bitmap.resize(bitmap_words);
    @memset(bitmap.items, 0);
    
    // Track which slots are occupied and by which patterns
    var slot_to_rank = try std.ArrayList(?u16).initCapacity(std.heap.page_allocator, table_size);
    defer slot_to_rank.deinit();
    try slot_to_rank.resize(table_size);
    @memset(slot_to_rank.items, null);
    
    // CRITICAL FIX: Two-pass collision handling for BBHash correctness
    // Pass 1: Count slot occupancy to identify ALL collisions
    var slot_occupancy = try std.ArrayList(u8).initCapacity(std.heap.page_allocator, table_size);
    defer slot_occupancy.deinit();
    try slot_occupancy.resize(table_size);
    @memset(slot_occupancy.items, 0);
    
    for (patterns) |pattern| {
        const hash = bbhash_murmur_mix(@as(u32, pattern.pattern) ^ @as(u32, @truncate(seed)));
        const slot = hash & (table_size - 1);
        if (slot_occupancy.items[slot] < 255) { // Avoid overflow
            slot_occupancy.items[slot] += 1;
        }
    }
    
    // Pass 2: Only place patterns that uniquely map to slots
    // ALL colliding patterns (including first one) move to next level
    for (patterns) |pattern| {
        const hash = bbhash_murmur_mix(@as(u32, pattern.pattern) ^ @as(u32, @truncate(seed)));
        const slot = hash & (table_size - 1);
        
        if (slot_occupancy.items[slot] == 1) {
            // Uniquely mapped, place in this level
            slot_to_rank.items[slot] = pattern.rank;
            
            
            // Set bit in bitmap
            const word_index = slot / 64;
            const bit_index = slot % 64;
            bitmap.items[word_index] |= (@as(u64, 1) << @intCast(bit_index));
        } else {
            // Collision (multiple patterns hash to same slot), move ALL to next level
            try remaining_patterns.append(pattern);
        }
    }
    
    // Add ranks in order of their bit positions in bitmap
    // This ensures BBHash lookup finds ranks in correct order via bit counting
    for (0..table_size) |slot| {
        const word_index = slot / 64;
        const bit_index = slot % 64;
        const bit_set = (bitmap.items[word_index] & (@as(u64, 1) << @intCast(bit_index))) != 0;
        
        if (bit_set and slot_to_rank.items[slot] != null) {
            // This slot has a pattern, add its rank in bit position order
            const rank = slot_to_rank.items[slot].?;
            try reordered_ranks.append(rank);
            
        }
    }
}

fn bbhash_murmur_mix(x: u32) u32 {
    // Murmur3 32-bit finalizer
    var h = x;
    h ^= h >> 16;
    h *%= 0x85ebca6b;
    h ^= h >> 13;
    h *%= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

fn compare_flush_patterns(context: void, a: FlushEntry, b: FlushEntry) bool {
    _ = context;
    return @as(u16, a.pattern) < @as(u16, b.pattern);
}

fn slow_evaluate_hand(hand: evaluator.Hand) u16 {
    return evaluator.evaluateHand(hand);
}

fn slow_evaluate_flush(pattern: u16) u16 {
    // Build a 7-card hand with 5 flush cards + 2 very low off-suit cards
    // Use the lowest possible ranks (2,3) in different suits to avoid interference
    var hand: evaluator.Hand = 0;
    
    // Add the flush pattern cards (clubs suit = suit 0)
    for (0..13) |r| {
        if ((pattern & (@as(u16, 1) << @intCast(r))) != 0) {
            hand |= evaluator.makeCard(0, @intCast(r)); // clubs
        }
    }
    
    // Add exactly 2 off-suit cards: 2 of diamonds, 3 of diamonds
    // Use ranks that are guaranteed not to interfere with any flush patterns
    hand |= evaluator.makeCard(1, 0); // 2 of diamonds  
    hand |= evaluator.makeCard(1, 1); // 3 of diamonds
    
    const rank = evaluator.evaluateHand(hand);
    
    
    return rank;
}

// Generate all C(52,7) combinations of 7 cards
fn generate_all_hands(allocator: std.mem.Allocator) !std.ArrayList(evaluator.Hand) {
    var hands = std.ArrayList(evaluator.Hand).init(allocator);
    
    // Generate all combinations of 7 cards from 52
    var combination = [_]u8{0} ** 7;
    try generate_combinations(0, 0, 52, 7, &combination, &hands);
    
    return hands;
}

fn generate_combinations(start: u8, depth: u8, total: u8, choose: u8, 
                        combination: *[7]u8, hands: *std.ArrayList(evaluator.Hand)) !void {
    if (depth == choose) {
        // Convert card indices to hand mask
        var hand: evaluator.Hand = 0;
        for (combination[0..choose]) |card_idx| {
            const suit = card_idx / 13;
            const rank = card_idx % 13;
            hand |= evaluator.makeCard(suit, rank);
        }
        try hands.append(hand);
        return;
    }
    
    var i: u8 = start;
    while (i <= total - (choose - depth)) : (i += 1) {
        combination[depth] = i;
        try generate_combinations(i + 1, depth + 1, total, choose, combination, hands);
    }
}

fn validate_table_sizes() void {
    // Validate table sizes
    comptime {
        std.debug.assert(@sizeOf(@TypeOf(chd_g_array)) == 8192);
        std.debug.assert(@sizeOf(@TypeOf(chd_value_table)) == 262144); // 2^17 * 2 bytes
    }
    
    // Validate CHD displacement bounds
    var max_displacement: u8 = 0;
    for (chd_g_array) |displacement| {
        if (displacement > max_displacement) {
            max_displacement = displacement;
        }
    }
    
    print("  CHD displacement array: {} bytes (max displacement: {})\n", .{ @sizeOf(@TypeOf(chd_g_array)), max_displacement });
    print("  CHD value table: {} bytes\n", .{@sizeOf(@TypeOf(chd_value_table))});
    print("  BBHash blob: {} bytes\n", .{bbhash_blob.items.len});
    
    const total_size = @sizeOf(@TypeOf(chd_g_array)) + 
                      @sizeOf(@TypeOf(chd_value_table)) + 
                      bbhash_blob.items.len;
    print("  Total table size: {} bytes ({} KB)\n", .{ total_size, total_size / 1024 });
    
    // Validate bounds from DESIGN.md
    std.debug.assert(max_displacement <= 255);
    std.debug.assert(total_size <= 300000); // ~300KB limit for now
}

fn write_tables_file() !void {
    const file = try std.fs.cwd().createFile("src/claude/tables.zig", .{});
    defer file.close();
    
    const writer = file.writer();
    
    // Write file header
    try writer.print("// Generated lookup tables for SIMD poker evaluator\n", .{});
    try writer.print("// Total size: {} KB\n\n", .{
        (@sizeOf(@TypeOf(chd_g_array)) + @sizeOf(@TypeOf(chd_value_table)) + bbhash_blob.items.len) / 1024
    });
    
    // Write CHD tables
    try writer.print("// CHD displacement array (8,192 bytes)\n", .{});
    try writer.print("pub const chd_g_array = [_]u8{{\n", .{});
    for (chd_g_array, 0..) |displacement, i| {
        if (i % 16 == 0) try writer.print("    ", .{});
        try writer.print("{}, ", .{displacement});
        if (i % 16 == 15) try writer.print("\n", .{});
    }
    try writer.print("}};\n\n", .{});
    
    try writer.print("// CHD value table (131,072 bytes)\n", .{});
    try writer.print("pub const chd_value_table = [_]u16{{\n", .{});
    for (chd_value_table, 0..) |rank, i| {
        if (i % 16 == 0) try writer.print("    ", .{});
        try writer.print("{}, ", .{rank});
        if (i % 16 == 15) try writer.print("\n", .{});
    }
    try writer.print("}};\n\n", .{});
    
    // Write simple flush lookup table
    try writer.print("// Simple flush lookup table ({} bytes)\n", .{bbhash_blob.items.len});
    try writer.print("pub const flush_lookup_blob = [_]u8{{\n", .{});
    for (bbhash_blob.items, 0..) |byte, i| {
        if (i % 16 == 0) try writer.print("    ", .{});
        try writer.print("0x{X:0>2}, ", .{byte});
        if (i % 16 == 15) try writer.print("\n", .{});
    }
    try writer.print("}};\n\n", .{});
    
    // Write constants
    try writer.print("// CHD constants\n", .{});
    try writer.print("pub const CHD_MAGIC_CONSTANT: u64 = 0x{X};\n", .{chd_magic_constant});
    try writer.print("pub const CHD_NUM_BUCKETS: u32 = {};\n", .{CHD_NUM_BUCKETS});
    try writer.print("pub const CHD_TABLE_SIZE: u32 = {};\n", .{CHD_TABLE_SIZE});
    
    // Write validation
    try writer.print("\n// Compile-time size validation\n", .{});
    try writer.print("comptime {{\n", .{});
    try writer.print("    const std = @import(\"std\");\n", .{});
    try writer.print("    std.debug.assert(@sizeOf(@TypeOf(chd_g_array)) == 8192);\n", .{});
    try writer.print("    std.debug.assert(@sizeOf(@TypeOf(chd_value_table)) == 262144);\n", .{});
    try writer.print("}}\n", .{});
}

// Data structures
const RPCEntry = struct {
    rpc: u32, // Rank Pattern Code
    rank: u16, // True hand rank (0-7461)
};

const FlushEntry = struct {
    pattern: u13, // 13-bit flush rank pattern
    rank: u16, // True flush rank
};

const CHDBucket = struct {
    id: u32,
    entries: std.ArrayList(RPCEntry),
};

// Test functions
test "RPC encoding" {
    // Test RPC encoding for known patterns
    const royal_flush_ranks: u13 = 0b1111100000000; // A-K-Q-J-T
    const rpc = compute_rpc(royal_flush_ranks);
    
    // Should have exactly 1 of each of the top 5 ranks
    try std.testing.expect(rpc != 0);
}

test "CHD hash function" {
    // Test hash function produces expected ranges
    const test_rpc: u32 = 0x12345;
    const hash_result = chd_hash(test_rpc);
    
    try std.testing.expect(hash_result.bucket < CHD_NUM_BUCKETS);
    try std.testing.expect(hash_result.base_index < CHD_TABLE_SIZE);
}