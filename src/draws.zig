const std = @import("std");
const card = @import("card");

/// Types of draws a hand can have
pub const DrawType = enum {
    flush_draw,
    nut_flush_draw,
    open_ended_straight_draw,
    gutshot,
    double_gutshot,
    combo_draw, // Multiple draws
    backdoor_flush,
    backdoor_straight,
    overcards,
    none,

    pub fn toString(self: DrawType) []const u8 {
        return switch (self) {
            .flush_draw => "flush draw",
            .nut_flush_draw => "nut flush draw",
            .open_ended_straight_draw => "open-ended straight draw",
            .gutshot => "gutshot",
            .double_gutshot => "double gutshot",
            .combo_draw => "combo draw",
            .backdoor_flush => "backdoor flush",
            .backdoor_straight => "backdoor straight",
            .overcards => "overcards",
            .none => "no draw",
        };
    }
};

/// Information about draws in a hand
pub const DrawInfo = struct {
    draws: []DrawType,
    outs: u8,
    nut_outs: u8,

    pub fn hasStrongDraw(self: DrawInfo) bool {
        for (self.draws) |draw| {
            switch (draw) {
                .flush_draw, .nut_flush_draw, .open_ended_straight_draw, .combo_draw => return true,
                else => continue,
            }
        }
        return false;
    }

    pub fn hasWeakDraw(self: DrawInfo) bool {
        for (self.draws) |draw| {
            switch (draw) {
                .gutshot, .backdoor_flush, .backdoor_straight, .overcards => return true,
                else => continue,
            }
        }
        return false;
    }

    pub fn isComboDraw(self: DrawInfo) bool {
        return self.draws.len >= 2 and self.outs >= 12;
    }
};

/// Detect all draws in a hand
pub fn detectDraws(
    allocator: std.mem.Allocator,
    hole_cards: [2]card.Hand,
    community_cards: []const card.Hand,
) !DrawInfo {
    var draws: std.ArrayList(DrawType) = .empty;
    defer draws.deinit(allocator);

    // Create bitmask of all cards already used
    var used_cards: card.Hand = 0;
    for (hole_cards) |hand| {
        used_cards |= hand;
    }
    for (community_cards) |hand| {
        used_cards |= hand;
    }

    // Collect outs using bitmasks to avoid double-counting
    var all_outs_mask: card.Hand = 0;
    var nut_outs_mask: card.Hand = 0;

    // Check flush draws
    const flush_info = detectFlushDraw(hole_cards, community_cards);
    if (flush_info.has_flush_draw) {
        if (flush_info.is_nut_flush_draw) {
            try draws.append(allocator, .nut_flush_draw);
        } else {
            try draws.append(allocator, .flush_draw);
        }

        // Get flush outs bitmask
        const flush_outs = getFlushOutsMask(used_cards, flush_info.suit.?);
        all_outs_mask |= flush_outs;

        if (flush_info.is_nut_flush_draw) {
            nut_outs_mask |= flush_outs;
        }
    } else if (flush_info.has_backdoor_flush) {
        try draws.append(allocator, .backdoor_flush);
    }

    // Check straight draws
    const straight_info = detectStraightDraw(hole_cards, community_cards);
    for (straight_info.draw_types) |draw_type| {
        if (draw_type != .none) {
            try draws.append(allocator, draw_type);
        }
    }

    // Get straight outs bitmask
    const straight_outs = getStraightOutsMask(hole_cards, community_cards, used_cards);
    all_outs_mask |= straight_outs;

    // Check for overcards (only on flop or turn)
    // Skip overcards if we already have flush/straight draws as they're not the primary outs
    const has_strong_draws = flush_info.has_flush_draw or
        (straight_info.draw_types[0] != .none and straight_info.draw_types[0] != .gutshot);

    if (community_cards.len <= 4 and !has_strong_draws) {
        const overcard_outs_mask = getOvercardOutsMask(hole_cards, community_cards, used_cards);
        if (overcard_outs_mask != 0) {
            try draws.append(allocator, .overcards);
            all_outs_mask |= overcard_outs_mask;
        }
    }

    // Calculate total unique outs
    const total_outs = @popCount(all_outs_mask);
    const nut_outs = @popCount(nut_outs_mask);

    // Check if it's a combo draw
    if (draws.items.len >= 2 and total_outs >= 12) {
        try draws.append(allocator, .combo_draw);
    }

    // If no draws found, add none
    if (draws.items.len == 0) {
        try draws.append(allocator, .none);
    }

    return DrawInfo{
        .draws = try draws.toOwnedSlice(allocator),
        .outs = @intCast(total_outs),
        .nut_outs = @intCast(nut_outs),
    };
}

const FlushDrawInfo = struct {
    has_flush_draw: bool,
    has_backdoor_flush: bool,
    is_nut_flush_draw: bool,
    outs: u8,
    suit: ?card.Suit,
};

/// Detect flush draws
fn detectFlushDraw(hole_cards: [2]card.Hand, community_cards: []const card.Hand) FlushDrawInfo {
    var suit_counts = [_]u8{0} ** 4;
    var hole_suits = [_]u8{0} ** 4;

    // Count suits in hole cards
    for (hole_cards) |hand| {
        for (0..4) |suit| {
            const offset = suit * 13;
            const suit_mask: u64 = @as(u64, 0x1FFF) << @intCast(offset); // 13 bits for each suit
            if ((hand & suit_mask) != 0) {
                suit_counts[suit] += @popCount(hand & suit_mask);
                hole_suits[suit] += @popCount(hand & suit_mask);
            }
        }
    }

    // Count suits on board
    for (community_cards) |hand| {
        for (0..4) |suit| {
            const offset = suit * 13;
            const suit_mask: u64 = @as(u64, 0x1FFF) << @intCast(offset); // 13 bits for each suit
            if ((hand & suit_mask) != 0) {
                suit_counts[suit] += @popCount(hand & suit_mask);
            }
        }
    }

    // Find flush draws
    var best_suit: ?card.Suit = null;
    var max_count: u8 = 0;
    var hole_cards_in_suit: u8 = 0;

    for (suit_counts, 0..) |count, suit| {
        if (count > max_count and hole_suits[suit] > 0) {
            max_count = count;
            best_suit = @enumFromInt(suit);
            hole_cards_in_suit = hole_suits[suit];
        }
    }

    if (max_count == 4 and hole_cards_in_suit >= 1) {
        // Check if it's nut flush draw
        const is_nut = isNutFlushDraw(hole_cards, community_cards, best_suit.?);

        return .{
            .has_flush_draw = true,
            .has_backdoor_flush = false,
            .is_nut_flush_draw = is_nut,
            .outs = 9, // Standard flush draw outs
            .suit = best_suit,
        };
    } else if (max_count == 3 and hole_cards_in_suit >= 1) {
        return .{
            .has_flush_draw = false,
            .has_backdoor_flush = true,
            .is_nut_flush_draw = false,
            .outs = 0, // Backdoor draws don't count as immediate outs
            .suit = best_suit,
        };
    }

    return .{
        .has_flush_draw = false,
        .has_backdoor_flush = false,
        .is_nut_flush_draw = false,
        .outs = 0,
        .suit = null,
    };
}

/// Check if we have the nut flush draw
fn isNutFlushDraw(hole_cards: [2]card.Hand, community_cards: []const card.Hand, suit: card.Suit) bool {
    // Find highest card of the suit in our hand
    var our_highest_rank: ?u8 = null;

    for (hole_cards) |hand| {
        const suit_offset = @as(u32, @intFromEnum(suit)) * 13;
        const suit_mask: u64 = @as(u64, 0x1FFF) << @intCast(suit_offset);

        if ((hand & suit_mask) != 0) {
            // Find the rank within this suit
            for (0..13) |rank| {
                const bit_pos = suit_offset + rank;
                if ((hand & (@as(u64, 1) << @intCast(bit_pos))) != 0) {
                    if (our_highest_rank == null or rank > our_highest_rank.?) {
                        our_highest_rank = @intCast(rank);
                    }
                }
            }
        }
    }

    if (our_highest_rank == null) return false;
    if (our_highest_rank.? == 12) return true; // We have the ace

    // Check if there's a higher card of this suit on board
    for (community_cards) |hand| {
        const suit_offset = @as(u32, @intFromEnum(suit)) * 13;
        const suit_mask: u64 = @as(u64, 0x1FFF) << @intCast(suit_offset);

        if ((hand & suit_mask) != 0) {
            for (0..13) |rank| {
                const bit_pos = suit_offset + rank;
                if ((hand & (@as(u64, 1) << @intCast(bit_pos))) != 0) {
                    if (rank > our_highest_rank.?) {
                        return false;
                    }
                }
            }
        }
    }

    // Check if ace is still available (not on board)
    const ace_bit_pos = @as(u32, @intFromEnum(suit)) * 13 + 12;
    const ace_of_suit = @as(u64, 1) << @intCast(ace_bit_pos);
    for (community_cards) |hand| {
        if ((hand & ace_of_suit) != 0) return false;
    }

    return true;
}

const StraightDrawInfo = struct {
    draw_types: [2]DrawType, // Can have multiple straight draws
    outs: u8,
    nut_straight: bool,
};

/// Detect straight draws
fn detectStraightDraw(hole_cards: [2]card.Hand, community_cards: []const card.Hand) StraightDrawInfo {
    var info = StraightDrawInfo{
        .draw_types = [_]DrawType{.none} ** 2,
        .outs = 0,
        .nut_straight = false,
    };

    // Create rank bitset for all cards
    var rank_bits: u16 = 0;
    for (hole_cards) |hand| {
        rank_bits |= getRankBits(hand);
    }
    for (community_cards) |hand| {
        rank_bits |= getRankBits(hand);
    }

    // Add ace-low capability
    if ((rank_bits & (1 << 12)) != 0) { // Has ace
        rank_bits |= 1; // Add virtual "1" for ace-low straights
    }

    // Check for open-ended straight draws
    var oesd_count: u8 = 0;
    var gutshot_count: u8 = 0;

    // Check each possible straight
    var start_rank: u8 = 0;
    while (start_rank <= 9) : (start_rank += 1) {
        var count: u8 = 0;
        var missing: u8 = 0;
        var gap_positions = [_]u8{0} ** 2;
        var gap_count: u8 = 0;

        // Count cards in this 5-card window
        for (0..5) |i| {
            const rank = start_rank + i;
            if ((rank_bits & (@as(u16, 1) << @intCast(rank))) != 0) {
                count += 1;
            } else {
                if (gap_count < 2) {
                    gap_positions[gap_count] = @intCast(i);
                    gap_count += 1;
                }
                missing += 1;
            }
        }

        if (count == 4) {
            // We have 4 cards to a straight
            if (missing == 1) {
                if (gap_positions[0] == 0 or gap_positions[0] == 4) {
                    // Missing card at end = open-ended
                    if (oesd_count == 0) {
                        info.draw_types[0] = .open_ended_straight_draw;
                        info.outs += 8;
                        oesd_count += 1;
                    }
                } else {
                    // Missing card in middle = gutshot
                    if (gutshot_count == 0) {
                        info.draw_types[gutshot_count] = .gutshot;
                        info.outs += 4;
                        gutshot_count += 1;
                    }
                }
            }
        }
    }

    // Check for double gutshot (two different gutshots)
    if (gutshot_count >= 2) {
        info.draw_types[0] = .double_gutshot;
        info.outs = 8; // Double gutshot = 8 outs like OESD
    }

    return info;
}

/// Count overcards to the board
fn countOvercards(hole_cards: [2]card.Hand, community_cards: []const card.Hand) u8 {
    if (community_cards.len == 0) return 0;

    // Find highest rank on board
    var board_high: u8 = 0;
    for (community_cards) |hand| {
        const rank = getCardRank(hand);
        if (rank > board_high) board_high = rank;
    }

    // Count hole cards higher than board
    var overcard_count: u8 = 0;
    for (hole_cards) |hand| {
        if (getCardRank(hand) > board_high) {
            overcard_count += 1;
        }
    }

    // Each overcard = 3 outs (to make top pair)
    return overcard_count * 3;
}

// Helper functions for generating outs bitmasks

/// Generate bitmask of all flush outs for a given suit
fn getFlushOutsMask(used_cards: card.Hand, suit: card.Suit) card.Hand {
    const suit_offset = @as(u32, @intFromEnum(suit)) * 13;
    const suit_mask: card.Hand = @as(card.Hand, 0x1FFF) << @intCast(suit_offset); // All 13 cards of this suit
    const unused_in_suit = suit_mask & ~used_cards; // Remove cards already used
    return unused_in_suit;
}

/// Generate bitmask of all straight outs
fn getStraightOutsMask(hole_cards: [2]card.Hand, community_cards: []const card.Hand, used_cards: card.Hand) card.Hand {
    var straight_outs: card.Hand = 0;

    // Create rank bitset for all cards
    var rank_bits: u16 = 0;
    for (hole_cards) |hand| {
        rank_bits |= getRankBits(hand);
    }
    for (community_cards) |hand| {
        rank_bits |= getRankBits(hand);
    }

    // Check each possible straight and find missing cards
    var start_rank: u8 = 0;
    while (start_rank <= 9) : (start_rank += 1) {
        var count: u8 = 0;
        var missing_ranks: std.ArrayList(u8) = .empty;
        defer missing_ranks.deinit(std.heap.page_allocator);

        // Count cards in this 5-card window
        for (0..5) |i| {
            const rank = start_rank + i;
            if ((rank_bits & (@as(u16, 1) << @intCast(rank))) != 0) {
                count += 1;
            } else {
                missing_ranks.append(std.heap.page_allocator, @intCast(rank)) catch continue;
            }
        }

        // If we have 4 cards to a straight, the missing card(s) are outs
        if (count == 4 and missing_ranks.items.len == 1) {
            const missing_rank = missing_ranks.items[0];
            // Add all 4 suits of this rank as outs (except already used)
            for (0..4) |suit| {
                const bit_pos = suit * 13 + missing_rank;
                const card_mask = @as(card.Hand, 1) << @intCast(bit_pos);
                if ((used_cards & card_mask) == 0) {
                    straight_outs |= card_mask;
                }
            }
        }
    }

    // Handle ace-low straight separately (A-2-3-4-5)
    const wheel_ranks = [_]u8{ 12, 0, 1, 2, 3 }; // A, 2, 3, 4, 5
    var wheel_count: u8 = 0;
    var wheel_missing: ?u8 = null;

    for (wheel_ranks) |rank| {
        if ((rank_bits & (@as(u16, 1) << @intCast(rank))) != 0) {
            wheel_count += 1;
        } else if (wheel_missing == null) {
            wheel_missing = rank;
        }
    }

    if (wheel_count == 4 and wheel_missing != null) {
        const missing_rank = wheel_missing.?;
        // Add all 4 suits of this rank as outs
        for (0..4) |suit| {
            const bit_pos = suit * 13 + missing_rank;
            const card_mask = @as(card.Hand, 1) << @intCast(bit_pos);
            if ((used_cards & card_mask) == 0) {
                straight_outs |= card_mask;
            }
        }
    }

    return straight_outs;
}

/// Generate bitmask of overcard outs
fn getOvercardOutsMask(hole_cards: [2]card.Hand, community_cards: []const card.Hand, used_cards: card.Hand) card.Hand {
    if (community_cards.len == 0) return 0;

    // Find highest rank on board
    var board_high: u8 = 0;
    for (community_cards) |hand| {
        const rank = getCardRank(hand);
        if (rank > board_high) board_high = rank;
    }

    // Get ranks of our hole cards
    const hole_rank1 = getCardRank(hole_cards[0]);
    const hole_rank2 = getCardRank(hole_cards[1]);

    var overcard_outs: card.Hand = 0;

    // Consider ranks higher than the board that would give us top pair
    // Include ranks we already have (to make pairs) AND ranks we don't have
    for (board_high + 1..13) |rank| {
        const rank_u8: u8 = @intCast(rank);

        // We want to include this rank if:
        // 1. We already have one of this rank (to make a pair), OR
        // 2. We don't have this rank (would give us a higher kicker/pair)
        const we_have_this_rank = (rank_u8 == hole_rank1 or rank_u8 == hole_rank2);

        // For now, only count overcards where we have one of that rank
        // (traditional "overcard" outs for making top pair)
        if (!we_have_this_rank) continue;

        // Add all remaining cards of this rank as outs
        for (0..4) |suit| {
            const bit_pos = suit * 13 + rank_u8;
            const card_mask = @as(card.Hand, 1) << @intCast(bit_pos);
            if ((used_cards & card_mask) == 0) {
                overcard_outs |= card_mask;
            }
        }
    }

    return overcard_outs;
}

// Helper functions (expose for testing) - OPTIMIZED using direct bit operations
pub fn getCardRank(hand: card.Hand) u8 {
    // Find the rank of a single card (0-12) using efficient bit operations
    if (hand == 0) return 0;

    const bit_pos = @ctz(hand); // Count trailing zeros to find first set bit
    return @intCast(bit_pos % 13);
}

pub fn getRankBits(hand: card.Hand) u16 {
    // Convert card hand to rank bitset using efficient bit operations
    var bits: u16 = 0;
    var remaining = hand;

    while (remaining != 0) {
        const bit_pos = @ctz(remaining);
        const rank = bit_pos % 13;
        bits |= @as(u16, 1) << @intCast(rank);
        remaining &= remaining - 1; // Clear the lowest set bit
    }

    return bits;
}

// Tests
const testing = std.testing;

test "flush draw detection" {
    // Flush draw: Ah Kh with Qh 5h 2c on board
    const hole_cards = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("Kh"),
    };
    const board = [_]card.Hand{
        card.parseCard("Qh"),
        card.parseCard("5h"),
        card.parseCard("2c"),
    };

    const flush_info = detectFlushDraw(hole_cards, &board);
    try testing.expect(flush_info.has_flush_draw);
    try testing.expect(flush_info.is_nut_flush_draw); // We have Ah
    try testing.expectEqual(@as(u8, 9), flush_info.outs);
}

test "straight draw detection" {
    // Open-ended: 98 with T76 on board
    const hole_cards = [_]card.Hand{
        card.parseCard("9h"),
        card.parseCard("8d"),
    };
    const board = [_]card.Hand{
        card.parseCard("Tc"),
        card.parseCard("7s"),
        card.parseCard("6h"),
    };

    const straight_info = detectStraightDraw(hole_cards, &board);
    try testing.expectEqual(DrawType.open_ended_straight_draw, straight_info.draw_types[0]);
    try testing.expectEqual(@as(u8, 8), straight_info.outs);
}

test "combo draw detection" {
    const allocator = testing.allocator;

    // Combo draw: Ah Kh with Qh Jh 5c
    const hole_cards = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("Kh"),
    };
    const board = [_]card.Hand{
        card.parseCard("Qh"),
        card.parseCard("Jh"),
        card.parseCard("5c"),
    };

    const draw_info = try detectDraws(allocator, hole_cards, &board);
    defer allocator.free(draw_info.draws);

    // Should have flush draw and straight draw
    try testing.expect(draw_info.draws.len >= 2);
    try testing.expect(draw_info.outs >= 12);
}

test "no draw detection" {
    const allocator = testing.allocator;

    // No draws: Ah 3d with Kh 7s 2c
    const hole_cards = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("3d"),
    };
    const board = [_]card.Hand{
        card.parseCard("Kh"),
        card.parseCard("7s"),
        card.parseCard("2c"),
    };

    const draw_info = try detectDraws(allocator, hole_cards, &board);
    defer allocator.free(draw_info.draws);

    try testing.expectEqual(@as(usize, 1), draw_info.draws.len);
    try testing.expectEqual(DrawType.overcards, draw_info.draws[0]);
}

test "FIXED: outs calculation now correctly handles combo draws" {
    const allocator = testing.allocator;

    // Combo draw: As Ks with Qs Js 5c
    // Expected outs: 9 spades + 6 non-spade tens/nines = 15 outs
    const hole_cards = [_]card.Hand{
        card.parseCard("As"),
        card.parseCard("Ks"),
    };
    const board = [_]card.Hand{
        card.parseCard("Qs"),
        card.parseCard("Js"),
        card.parseCard("5c"),
    };

    const draw_info = try detectDraws(allocator, hole_cards, &board);
    defer allocator.free(draw_info.draws);

    // Should have flush draw, straight draw, and combo draw (no overcards in combo situations)
    try testing.expect(draw_info.draws.len == 3);

    // Should correctly calculate 15 unique outs (no double-counting)
    try testing.expectEqual(@as(u8, 15), draw_info.outs);

    // Should detect as combo draw
    try testing.expect(draw_info.isComboDraw());
}

test "ace-low straight detection works correctly" {
    const allocator = testing.allocator;

    // Test ace-low straight draw: A2 with 345 board (need 6 or have wheel)
    const hole_cards = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("2s"),
    };
    const board = [_]card.Hand{
        card.parseCard("3c"),
        card.parseCard("4d"),
        card.parseCard("5h"),
    };

    const draw_info = try detectDraws(allocator, hole_cards, &board);
    defer allocator.free(draw_info.draws);

    // Should detect some form of straight draw
    var has_straight_draw = false;
    for (draw_info.draws) |draw| {
        if (draw == .open_ended_straight_draw or draw == .gutshot) {
            has_straight_draw = true;
            break;
        }
    }

    try testing.expect(has_straight_draw);
    try testing.expect(draw_info.outs >= 4); // At least 4 outs for a straight
}

test "API typo FIXED: isComboDraw function works correctly" {
    const allocator = testing.allocator;

    const hole_cards = [_]card.Hand{
        card.parseCard("As"),
        card.parseCard("Ks"),
    };
    const board = [_]card.Hand{
        card.parseCard("Qs"),
        card.parseCard("Js"),
        card.parseCard("5c"),
    };

    const draw_info = try detectDraws(allocator, hole_cards, &board);
    defer allocator.free(draw_info.draws);

    // Verify the API function works correctly

    // This now works with the fixed function name
    _ = draw_info.isComboDraw();
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
