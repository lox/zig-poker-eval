const std = @import("std");
const card = @import("card");

/// Board texture classification from dry to very wet
pub const BoardTexture = enum {
    dry,
    semi_wet,
    wet,
    very_wet,

    pub fn toString(self: BoardTexture) []const u8 {
        return switch (self) {
            .dry => "dry",
            .semi_wet => "semi-wet",
            .wet => "wet",
            .very_wet => "very wet",
        };
    }
};

/// Information about flush potential on a board
pub const FlushInfo = struct {
    max_suit_count: u8,
    dominant_suit: ?card.Suit,
    is_monotone: bool,
    is_rainbow: bool,
};

/// Information about straight potential on a board
pub const StraightInfo = struct {
    connected_cards: u8,
    gaps: u8,
    has_ace: bool,
    broadway_cards: u8, // TJQKA
};

/// Analyze how coordinated/dangerous a board is
pub fn analyzeBoardTexture(board: []const card.Hand) BoardTexture {
    if (board.len < 3) return .dry;

    var wetness: u32 = 0;

    // Check flush possibilities
    const flush_info = analyzeFlushPotential(board);
    if (flush_info.max_suit_count >= 3) {
        wetness += 2; // Flush draw possible
    } else if (flush_info.max_suit_count == 2) {
        wetness += 1; // Two-suited
    }

    // Check straight possibilities
    const straight_info = analyzeStraightPotential(board);
    if (straight_info.connected_cards >= 3) {
        wetness += 2; // Straight draws likely
    } else if (straight_info.connected_cards == 2) {
        wetness += 1; // Some connectivity
    }

    // Check for pairs on board
    const pair_count = countBoardPairs(board);
    if (pair_count >= 1) {
        wetness += 1; // Paired board
    }

    // High card concentration (multiple high cards = more dangerous)
    const high_card_count = countHighCards(board);
    if (high_card_count >= 3) {
        wetness += 1;
    }

    return switch (wetness) {
        0, 1 => .dry,
        2, 3 => .semi_wet,
        4, 5 => .wet,
        else => .very_wet,
    };
}

/// Analyze flush potential on the board
pub fn analyzeFlushPotential(board: []const card.Hand) FlushInfo {
    var suit_counts = [_]u8{0} ** 4;

    // Count cards of each suit
    for (board) |hand| {
        // Count bits set in each suit's range
        for (0..4) |suit| {
            const offset = suit * 13;
            const suit_mask: u64 = @as(u64, 0x1FFF) << @intCast(offset); // 13 bits for each suit
            const suit_cards = hand & suit_mask;
            if (suit_cards != 0) {
                suit_counts[suit] += @popCount(suit_cards);
            }
        }
    }

    var max_count: u8 = 0;
    var dominant_suit: ?card.Suit = null;
    var non_zero_suits: u8 = 0;

    for (suit_counts, 0..) |count, suit| {
        if (count > 0) {
            non_zero_suits += 1;
            if (count > max_count) {
                max_count = count;
                dominant_suit = @enumFromInt(suit);
            }
        }
    }

    return .{
        .max_suit_count = max_count,
        .dominant_suit = dominant_suit,
        .is_monotone = (non_zero_suits == 1 and board.len >= 3),
        .is_rainbow = (non_zero_suits == board.len and board.len >= 3),
    };
}

/// Analyze straight potential on the board
pub fn analyzeStraightPotential(board: []const card.Hand) StraightInfo {
    if (board.len < 2) return .{ .connected_cards = 1, .gaps = 0, .has_ace = false, .broadway_cards = 0 };

    // Get unique ranks on board
    var rank_set: u16 = 0; // Bit set for ranks (0-12)
    var has_ace = false;
    var broadway_count: u8 = 0;

    for (board) |hand| {
        // Check each rank across all suits
        for (0..13) |rank| {
            var rank_mask: u64 = 0;
            // Build mask for this rank across all suits
            for (0..4) |suit| {
                const bit_pos = suit * 13 + rank;
                rank_mask |= @as(u64, 1) << @intCast(bit_pos);
            }

            if ((hand & rank_mask) != 0) {
                rank_set |= @as(u16, 1) << @intCast(rank);
                if (rank == 12) has_ace = true; // Ace
                if (rank >= 8) broadway_count += 1; // Ten or higher
            }
        }
    }

    // Count longest sequence of connected ranks
    var max_connected: u8 = 0;
    var current_connected: u8 = 0;
    var total_gaps: u8 = 0;
    var last_rank: i32 = -1;

    for (0..13) |rank| {
        if ((rank_set & (@as(u16, 1) << @intCast(rank))) != 0) {
            if (last_rank >= 0) {
                const gap = @as(u8, @intCast(rank)) - @as(u8, @intCast(last_rank)) - 1;
                if (gap <= 1) {
                    current_connected += 1;
                    total_gaps += gap;
                } else {
                    if (current_connected > max_connected) {
                        max_connected = current_connected;
                    }
                    current_connected = 1;
                }
            } else {
                current_connected = 1;
            }
            last_rank = @as(i32, @intCast(rank));
        }
    }

    // Check for ace-low straight potential (A-2-3-4-5)
    if (has_ace and (rank_set & 0x0F) != 0) { // Has ace and at least one of 2-5
        current_connected += 1;
    }

    if (current_connected > max_connected) {
        max_connected = current_connected;
    }

    return .{
        .connected_cards = max_connected,
        .gaps = total_gaps,
        .has_ace = has_ace,
        .broadway_cards = broadway_count,
    };
}

/// Count pairs on the board
pub fn countBoardPairs(board: []const card.Hand) u8 {
    var rank_counts = [_]u8{0} ** 13;

    for (board) |hand| {
        for (0..13) |rank| {
            var rank_mask: u64 = 0;
            // Build mask for this rank across all suits
            for (0..4) |suit| {
                const bit_pos = suit * 13 + rank;
                rank_mask |= @as(u64, 1) << @intCast(bit_pos);
            }

            if ((hand & rank_mask) != 0) {
                rank_counts[rank] += 1;
            }
        }
    }

    var pairs: u8 = 0;
    for (rank_counts) |count| {
        if (count >= 2) pairs += 1;
    }

    return pairs;
}

/// Count high cards (Ten or higher) on board
pub fn countHighCards(board: []const card.Hand) u8 {
    var count: u8 = 0;

    for (board) |hand| {
        // Check for T, J, Q, K, A (ranks 8-12)
        for (8..13) |rank| {
            var rank_mask: u64 = 0;
            // Build mask for this rank across all suits
            for (0..4) |suit| {
                const bit_pos = suit * 13 + rank;
                rank_mask |= @as(u64, 1) << @intCast(bit_pos);
            }

            if ((hand & rank_mask) != 0) {
                count += 1;
            }
        }
    }

    return count;
}

/// Check if board is monotone (all same suit)
pub fn isMonotone(board: []const card.Hand) bool {
    if (board.len < 3) return false;
    const info = analyzeFlushPotential(board);
    return info.is_monotone;
}

/// Check if board is rainbow (all different suits)
pub fn isRainbow(board: []const card.Hand) bool {
    if (board.len < 3) return false;
    const info = analyzeFlushPotential(board);
    return info.is_rainbow;
}

/// Check if board has three or more to a flush
pub fn hasFlushDraw(board: []const card.Hand) bool {
    const info = analyzeFlushPotential(board);
    return info.max_suit_count >= 3;
}

/// Check if board has three or more connected cards
pub fn hasStraightDraw(board: []const card.Hand) bool {
    const info = analyzeStraightPotential(board);
    return info.connected_cards >= 3;
}

/// Check if board is paired
pub fn isPairedBoard(board: []const card.Hand) bool {
    return countBoardPairs(board) > 0;
}

// Tests
const testing = std.testing;

test "board texture analysis - dry board" {
    // Ah 7d 2c - rainbow, no connectivity
    const board = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("7d"),
        card.parseCard("2c"),
    };

    const texture = analyzeBoardTexture(&board);
    try testing.expectEqual(BoardTexture.dry, texture);
    try testing.expect(isRainbow(&board));
    try testing.expect(!hasFlushDraw(&board));
    try testing.expect(!hasStraightDraw(&board));
}

test "board texture analysis - wet board" {
    // Qh Jh Th - monotone, highly connected
    const board = [_]card.Hand{
        card.parseCard("Qh"),
        card.parseCard("Jh"),
        card.parseCard("Th"),
    };

    const texture = analyzeBoardTexture(&board);
    try testing.expectEqual(BoardTexture.wet, texture);
    try testing.expect(isMonotone(&board));
    try testing.expect(hasFlushDraw(&board));
    try testing.expect(hasStraightDraw(&board));
}

test "board texture analysis - paired board" {
    // Kh Kd 7c
    const board = [_]card.Hand{
        card.parseCard("Kh"),
        card.parseCard("Kd"),
        card.parseCard("7c"),
    };

    try testing.expect(isPairedBoard(&board));
    try testing.expectEqual(@as(u8, 1), countBoardPairs(&board));
}

test "flush potential analysis" {
    // Three hearts
    const board = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("Kh"),
        card.parseCard("5h"),
        card.parseCard("2d"),
    };

    const info = analyzeFlushPotential(&board);
    try testing.expectEqual(@as(u8, 3), info.max_suit_count);
    try testing.expectEqual(card.Suit.hearts, info.dominant_suit.?);
    try testing.expect(!info.is_monotone);
    try testing.expect(!info.is_rainbow);
}

test "straight potential analysis" {
    // 9-T-J-Q
    const board = [_]card.Hand{
        card.parseCard("9h"),
        card.parseCard("Td"),
        card.parseCard("Jc"),
        card.parseCard("Qs"),
    };

    const info = analyzeStraightPotential(&board);
    try testing.expectEqual(@as(u8, 4), info.connected_cards);
    try testing.expectEqual(@as(u8, 3), info.broadway_cards);
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
