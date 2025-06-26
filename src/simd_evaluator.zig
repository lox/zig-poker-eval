const std = @import("std");
const tables = @import("tables.zig");
const slow_evaluator = @import("slow_evaluator.zig");

// SIMD vector types - only expose what's actually used externally
pub const VecU64 = @Vector(16, u64);

// Internal vector types
const VecU32 = @Vector(16, u32);
const VecU16 = @Vector(16, u16);

// Evaluator configuration
const BATCH_SIZE = 16;
const RANK_MASK = 0x1FFF; // 13 bits for ranks A-K-Q-...-2
const SUIT_SHIFT = [4]u6{ 0, 13, 26, 39 }; // Club, Diamond, Heart, Spade bit positions

pub const SIMDEvaluator = struct {
    const Self = @This();

    pub fn init() Self {
        return Self{};
    }

    /// Evaluate a batch of 16 hands simultaneously
    pub fn evaluate_batch(self: *const Self, hands: VecU64) VecU16 {
        // Split card masks into rank and suit components
        const masks = self.split_card_masks(hands);

        // Detect flush lanes
        const flush_info = self.detect_flush_lanes(masks.suits);

        // Evaluate non-flush hands (majority case)
        const non_flush_ranks = self.evaluate_non_flush_path(hands);

        // Evaluate flush hands for lanes that have flushes
        const flush_ranks = self.evaluate_flush_path(masks.suits, flush_info.predicate);

        // Select results based on flush predicate (branch-free)
        var results: VecU16 = non_flush_ranks;
        var predicate: VecU16 = @splat(0);

        for (0..16) |i| {
            if (flush_info.predicate[i] != 0) {
                predicate[i] = 0xFFFF;
            }
        }

        // Blend flush and non-flush results
        for (0..16) |i| {
            if (predicate[i] != 0) {
                results[i] = flush_ranks[i];
            }
        }

        return results;
    }

    // Internal helper functions
    fn split_card_masks(self: *const Self, hands: VecU64) struct { suits: [4]VecU64 } {
        _ = self;
        const mask = @as(VecU64, @splat(RANK_MASK));

        return .{
            .suits = [4]VecU64{
                hands & mask, // clubs
                (hands >> @as(VecU64, @splat(SUIT_SHIFT[1]))) & mask, // diamonds
                (hands >> @as(VecU64, @splat(SUIT_SHIFT[2]))) & mask, // hearts
                (hands >> @as(VecU64, @splat(SUIT_SHIFT[3]))) & mask, // spades
            },
        };
    }

    fn detect_flush_lanes(self: *const Self, suits: [4]VecU64) struct { predicate: VecU16 } {
        _ = self;
        var predicate: VecU16 = @splat(0);

        for (0..16) |i| {
            for (suits) |suit| {
                if (@popCount(suit[i]) >= 5) {
                    predicate[i] = 0xFFFF;
                    break;
                }
            }
        }

        return .{ .predicate = predicate };
    }

    fn evaluate_non_flush_path(self: *const Self, hands: VecU64) VecU16 {
        _ = self;
        var results: VecU16 = @splat(0);

        // For now, use scalar fallback for each hand
        for (0..16) |i| {
            const rpc = compute_rpc_from_hand(hands[i]);
            results[i] = chd_lookup_scalar(rpc);
        }

        return results;
    }

    fn evaluate_flush_path(self: *const Self, suits: [4]VecU64, predicate: VecU16) VecU16 {
        _ = self;
        var results: VecU16 = @splat(0);

        // For each lane that has a flush, evaluate it
        for (0..16) |i| {
            if (predicate[i] != 0) {
                results[i] = evaluate_flush_single(suits, i);
            }
        }

        return results;
    }
};

// Single hand evaluation - just delegates to slow evaluator for simplicity
pub fn evaluate_single_hand(hand: u64) u16 {
    return slow_evaluator.evaluateHand(hand);
}

// Helper functions that were used internally
fn compute_rpc_from_hand(hand: u64) u32 {
    // Extract rank counts for each of the 13 ranks
    var rank_counts = [_]u8{0} ** 13;

    for (0..4) |suit| {
        const suit_mask = @as(u16, @truncate((hand >> (@as(u6, @intCast(suit)) * 13)) & RANK_MASK));
        for (0..13) |rank| {
            if ((suit_mask & (@as(u16, 1) << @intCast(rank))) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    // Base-5 encoding: preserves all patterns in 31 bits
    var rpc: u32 = 0;
    for (rank_counts) |count| {
        rpc = rpc * 5 + count;
    }

    return rpc;
}

fn chd_lookup_scalar(rpc: u32) u16 {
    const h = mix64(@as(u64, rpc));
    const bucket = @as(u32, @intCast(h >> 51)); // Top 13 bits
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits
    const displacement = tables.chd_g_array[bucket];
    const final_index = (base_index + displacement) & (tables.CHD_TABLE_SIZE - 1);
    return tables.chd_value_table[final_index];
}

fn mix64(x: u64) u64 {
    var result = x;
    result ^= result >> 33;
    result *%= tables.CHD_MAGIC_CONSTANT;
    result ^= result >> 29;
    return result;
}

fn evaluate_flush_single(suits: [4]VecU64, lane: usize) u16 {
    // Find which suit has the flush
    for (0..4) |suit| {
        if (@popCount(suits[suit][lane]) >= 5) {
            const suit_mask = @as(u16, @truncate(suits[suit][lane]));

            // Check for straight flush first
            const straight_mask = get_straight_mask(suit_mask);
            if (straight_mask != 0) {
                if (straight_mask == 0x1F00) return 0; // Royal flush
                if (straight_mask == 0x100F) return 9; // Wheel (A-5)

                // Other straight flushes
                const high_bit = @clz(straight_mask);
                return @as(u16, 12 - (15 - high_bit));
            }

            // Regular flush
            const flush_pattern = get_top5_ranks(suit_mask);
            return flush_lookup_scalar(flush_pattern);
        }
    }

    return 0; // Should not reach here
}

fn get_straight_mask(ranks: u16) u16 {
    const straight_patterns = [_]u16{
        0x1F00, // A,K,Q,J,10 (royal)
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

    for (straight_patterns) |pattern| {
        if ((ranks & pattern) == pattern) {
            return pattern;
        }
    }

    return 0;
}

fn get_top5_ranks(suit_mask: u16) u16 {
    // Take highest 5 ranks
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

fn flush_lookup_scalar(pattern: u16) u16 {
    // Simplified flush lookup - just delegate to slow evaluator for now
    // Build a minimal hand with this flush pattern
    var hand: u64 = 0;
    for (0..13) |r| {
        if ((pattern & (@as(u16, 1) << @intCast(r))) != 0) {
            hand |= slow_evaluator.makeCard(0, @intCast(r)); // clubs
        }
    }

    // Add some non-conflicting cards to make it 7 cards
    hand |= slow_evaluator.makeCard(1, 0); // 2 of diamonds
    hand |= slow_evaluator.makeCard(1, 1); // 3 of diamonds

    return slow_evaluator.evaluateHand(hand);
}
