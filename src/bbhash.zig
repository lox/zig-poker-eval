//! BBHash (Bloom-based Hash) for flush pattern evaluation
//! 
//! Implements a minimal perfect hash for 7-card flush patterns.
//! Used in the flush evaluation path of the poker hand evaluator.
//!
//! Design based on DESIGN.md specification:
//! - Input: 13-bit flush ranks (suit mask with ≥5 cards)
//! - Hash: multiply by magic constant + shift
//! - Output: poker hand ranking from 8KB lookup table
//!
//! Handles:
//! - Straight flushes (including royal flush)
//! - Wheel straight flush (A-2-3-4-5)
//! - Regular flushes ranked by high card

const std = @import("std");
const print = std.debug.print;

/// BBHash result containing the magic constant and lookup table
pub const BBHashResult = struct {
    magic: u32,           // Magic multiplication constant
    shift: u6,            // Right shift amount  
    table_size: u32,      // Size of lookup table (power of 2)
    values: []u16,        // Lookup table: hash_index -> poker_rank
    
    pub fn deinit(self: BBHashResult, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
    }
};

/// Flush pattern for BBHash construction
pub const FlushPattern = struct {
    ranks: u16,    // 13-bit rank mask (suits with ≥5 cards)
    value: u16,    // Poker hand ranking
};

/// Hash function used by BBHash (multiply + shift)
pub fn bbHashFunction(key: u16, magic: u32, shift: u6) u32 {
    // Extend to 32-bit for multiplication, then extract high bits
    const extended_key = @as(u32, key);
    const product = extended_key *% magic;  // Wrapping multiplication
    return product >> @as(u5, @intCast(shift));
}

/// Find a perfect hash constant for the given flush patterns
pub fn findPerfectHashConstant(
    allocator: std.mem.Allocator,
    patterns: []const FlushPattern,
    table_size: u32
) !u32 {
    std.debug.assert(table_size > 0 and (table_size & (table_size - 1)) == 0); // Power of 2
    
    const shift = @as(u6, @intCast(32 - @ctz(table_size))); // log2(table_size)
    var used_slots = try allocator.alloc(bool, table_size);
    defer allocator.free(used_slots);
    
    // Try odd constants to stay coprime with powers of 2
    var magic: u32 = 1;
    while (magic < 0xFFFFFFFF) : (magic += 2) {
        @memset(used_slots, false);
        
        var collision = false;
        for (patterns) |pattern| {
            const idx = bbHashFunction(pattern.ranks, magic, shift);
            if (idx >= table_size) {
                collision = true;
                break;
            }
            if (used_slots[idx]) {
                collision = true;
                break;
            }
            used_slots[idx] = true;
        }
        
        if (!collision) {
            print("  BBHash: Found perfect constant 0x{x} with shift {} for {} patterns\n", 
                  .{ magic, shift, patterns.len });
            return magic;
        }
        
        // Progress indicator for long searches
        if (magic % 1000000 == 1) {
            print("  BBHash: Searching... tried {} constants\n", .{magic / 2});
        }
    }
    
    return error.NoValidConstantFound;
}

/// Build BBHash lookup table for flush patterns
pub fn buildBBHash(
    allocator: std.mem.Allocator,
    patterns: []const FlushPattern
) !BBHashResult {
    // Use 8192 entries (8KB with u16 values) as per DESIGN.md
    const table_size: u32 = 8192;
    const shift = @as(u6, @intCast(32 - @ctz(table_size))); // shift = 19
    
    print("  BBHash: Building hash for {} flush patterns\n", .{patterns.len});
    print("  BBHash: Table size {} entries, shift {}\n", .{ table_size, shift });
    
    // Find perfect hash constant
    const magic = try findPerfectHashConstant(allocator, patterns, table_size);
    
    // Allocate and populate lookup table
    var values = try allocator.alloc(u16, table_size);
    @memset(values, 0); // Initialize to 0 (invalid/unused slots)
    
    for (patterns) |pattern| {
        const idx = bbHashFunction(pattern.ranks, magic, shift);
        std.debug.assert(idx < table_size);
        std.debug.assert(values[idx] == 0); // No collisions
        values[idx] = pattern.value;
    }
    
    return BBHashResult{
        .magic = magic,
        .shift = shift,
        .table_size = table_size,
        .values = values,
    };
}

/// Runtime BBHash lookup
pub fn lookup(ranks: u16, magic: u32, shift: u6, values: []const u16) u16 {
    const idx = bbHashFunction(ranks, magic, shift);
    std.debug.assert(idx < values.len);
    return values[idx];
}

/// Extract the 13-bit rank mask for a suit with ≥5 cards
pub fn extractFlushRanks(hand: u64) ?u16 {
    // Extract 4 suit masks from 52-bit hand representation
    const suit_masks = [4]u16{
        @as(u16, @intCast((hand >> 0) & 0x1FFF)),  // Clubs
        @as(u16, @intCast((hand >> 13) & 0x1FFF)), // Diamonds  
        @as(u16, @intCast((hand >> 26) & 0x1FFF)), // Hearts
        @as(u16, @intCast((hand >> 39) & 0x1FFF)), // Spades
    };
    
    // Find suit with ≥5 cards
    for (suit_masks) |mask| {
        if (@popCount(mask) >= 5) {
            return mask;
        }
    }
    
    return null; // No flush
}

/// Determine poker ranking for flush ranks
pub fn evaluateFlushRanks(flush_ranks: u16) u16 {
    std.debug.assert(@popCount(flush_ranks) >= 5);
    
    // Check for straight flush first
    if (isStraightFlush(flush_ranks)) {
        if (isRoyalFlush(flush_ranks)) {
            return 7461; // Royal flush (best possible)
        } else if (isWheelStraightFlush(flush_ranks)) {
            return 7000; // Wheel straight flush (A-2-3-4-5)
        } else {
            // Regular straight flush - rank by high card
            const high_straight_card = getHighStraightCard(flush_ranks);
            return @as(u16, 7000) + @as(u16, high_straight_card); // 7001-7460
        }
    }
    
    // Regular flush - rank by high cards (top 5)
    const top5_ranks = getTop5Ranks(flush_ranks);
    return @as(u16, 5000) + rankFlushByHighCards(top5_ranks); // 5001-5999
}

/// Check if flush ranks form a straight flush
fn isStraightFlush(ranks: u16) bool {
    return isStraight(ranks);
}

/// Check if ranks form a straight (any 5 consecutive)
fn isStraight(ranks: u16) bool {
    // Check regular straights (5 consecutive bits)
    var i: u4 = 0;
    while (i <= 8) : (i += 1) {
        const straight_mask = @as(u16, 0x1F) << @as(u4, i); // 5 consecutive bits
        if ((ranks & straight_mask) == straight_mask) {
            return true;
        }
    }
    
    // Check wheel (A-2-3-4-5): bits 0,1,2,3,12
    const wheel_mask: u16 = 0x100F; // 0b1000000001111
    return (ranks & wheel_mask) == wheel_mask;
}

/// Check if flush ranks form a royal flush (A-K-Q-J-10)
fn isRoyalFlush(ranks: u16) bool {
    const royal_mask: u16 = 0x1F00; // Bits 8,9,10,11,12 (T,J,Q,K,A)
    return (ranks & royal_mask) == royal_mask;
}

/// Check if flush ranks form a wheel straight flush (A-2-3-4-5)
fn isWheelStraightFlush(ranks: u16) bool {
    const wheel_mask: u16 = 0x100F; // Bits 0,1,2,3,12 (2,3,4,5,A)
    return (ranks & wheel_mask) == wheel_mask;
}

/// Get the high card of a straight (non-wheel)
fn getHighStraightCard(ranks: u16) u8 {
    if (isWheelStraightFlush(ranks)) {
        return 3; // Wheel high card is 5 (index 3)
    }
    
    // Find highest bit in straight
    var i: u4 = 12;
    while (i >= 4) : (i -= 1) {
        const straight_mask = @as(u16, 0x1F) << @as(u4, (i - 4)); // 5 consecutive ending at i
        if ((ranks & straight_mask) == straight_mask) {
            return @as(u8, i);
        }
    }
    
    return 0; // Shouldn't happen if isStraight returned true
}

/// Get top 5 ranks from flush ranks (for regular flush evaluation)
fn getTop5Ranks(ranks: u16) u16 {
    var result = ranks;
    while (@popCount(result) > 5) {
        // Clear lowest set bit
        result = result & (result - 1);
    }
    return result;
}

/// Rank a regular flush by high cards (top 5)
fn rankFlushByHighCards(top5_ranks: u16) u16 {
    // Scale the ranking to fit in 999 possible values (5001-5999)
    // Use a simple hash of the bit pattern, scaled appropriately
    std.debug.assert(@popCount(top5_ranks) == 5);
    
    // For now, use a simple scaling. In a real implementation, this would
    // be a proper ranking system based on the actual card values.
    const scaled = @as(u32, top5_ranks) * 999 / 0x1FFF; // Scale to 0-999
    return @as(u16, @min(scaled, 999));
}

// ============================================================================
// TESTS
// ============================================================================

test "bbHashFunction basic" {
    const magic: u32 = 0x9E3779B9; // Golden ratio constant
    const shift: u6 = 19; // For 8192 table size
    
    // Test some basic inputs
    const idx1 = bbHashFunction(0x1F00, magic, shift); // Royal flush ranks
    const idx2 = bbHashFunction(0x100F, magic, shift); // Wheel ranks
    const idx3 = bbHashFunction(0x1E00, magic, shift); // K-high flush
    
    // Results should be in valid range
    try std.testing.expect(idx1 < 8192);
    try std.testing.expect(idx2 < 8192);
    try std.testing.expect(idx3 < 8192);
    
    // Different inputs should (probably) give different outputs
    try std.testing.expect(idx1 != idx2);
    try std.testing.expect(idx2 != idx3);
}

test "extractFlushRanks" {
    // Test hand with clubs flush: A♣ K♣ Q♣ J♣ 10♣ 9♣ 8♣
    const clubs_flush = 0x1F80; // Bits 7-12 in clubs suit (position 0-12)
    const hand = clubs_flush; // Only clubs, other suits empty
    
    const flush_ranks = extractFlushRanks(hand);
    try std.testing.expect(flush_ranks != null);
    try std.testing.expect(flush_ranks.? == clubs_flush);
    
    // Test hand with no flush - ensure each suit has <5 cards
    const no_flush_hand: u64 = 0x1001001001001; // 1 card per suit across different ranks
    const no_flush = extractFlushRanks(no_flush_hand);
    if (no_flush != null) {
        std.debug.print("Expected no flush but got: 0x{x}\n", .{no_flush.?});
    }
    try std.testing.expect(no_flush == null);
}

test "evaluateFlushRanks royal flush" {
    const royal_ranks: u16 = 0x1F00; // A-K-Q-J-10 (bits 8-12)
    const ranking = evaluateFlushRanks(royal_ranks);
    try std.testing.expect(ranking == 7461); // Best possible hand
}

test "evaluateFlushRanks wheel straight flush" {
    const wheel_ranks: u16 = 0x100F; // A-2-3-4-5 (bits 0,1,2,3,12)
    const ranking = evaluateFlushRanks(wheel_ranks);
    try std.testing.expect(ranking == 7000); // Wheel straight flush
}

test "evaluateFlushRanks regular straight flush" {
    const straight_ranks: u16 = 0x01F0; // 9-8-7-6-5 (bits 4-8)
    const ranking = evaluateFlushRanks(straight_ranks);
    try std.testing.expect(ranking >= 7001 and ranking < 7461); // Regular straight flush
}

test "evaluateFlushRanks regular flush" {
    const flush_ranks: u16 = 0x1E01; // A-K-Q-J-2 (non-straight)
    const ranking = evaluateFlushRanks(flush_ranks);
    std.debug.print("Regular flush ranking: {}\n", .{ranking});
    try std.testing.expect(ranking >= 5000 and ranking < 6000); // Regular flush
}

test "isStraight detection" {
    // Test wheel (A-2-3-4-5)
    try std.testing.expect(isStraight(0x100F));
    
    // Test regular straight (9-8-7-6-5)
    try std.testing.expect(isStraight(0x01F0));
    
    // Test royal (A-K-Q-J-10)
    try std.testing.expect(isStraight(0x1F00));
    
    // Test non-straight
    try std.testing.expect(!isStraight(0x1E01)); // A-K-Q-J-2
}

test "getTop5Ranks" {
    // Test with 7 cards, should keep top 5
    const seven_ranks: u16 = 0x1F03; // A-K-Q-J-10-3-2 (7 cards)
    const top5 = getTop5Ranks(seven_ranks);
    try std.testing.expect(@popCount(top5) == 5);
    try std.testing.expect((top5 & 0x1F00) == 0x1F00); // Should keep A-K-Q-J-10
}

test "BBHash integration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Create test patterns
    var patterns = [_]FlushPattern{
        .{ .ranks = 0x1F00, .value = 7461 }, // Royal flush
        .{ .ranks = 0x100F, .value = 7000 }, // Wheel straight flush
        .{ .ranks = 0x1E00, .value = 5500 }, // K-high flush
    };
    
    // Build BBHash
    const result = try buildBBHash(allocator, &patterns);
    defer result.deinit(allocator);
    
    // Test lookups
    try std.testing.expect(lookup(0x1F00, result.magic, result.shift, result.values) == 7461);
    try std.testing.expect(lookup(0x100F, result.magic, result.shift, result.values) == 7000);
    try std.testing.expect(lookup(0x1E00, result.magic, result.shift, result.values) == 5500);
}
