const std = @import("std");
const poker = @import("poker.zig");

/// Poker notation parsing for shorthand like AKo, AKs, 88, etc.
/// This extends basic card parsing to handle common poker shorthand notation.
pub const NotationError = error{
    InvalidNotation,
    InvalidRank,
    InvalidLength,
    UnsupportedNotation,
};

/// Parse poker shorthand notation into specific card combinations
/// Examples:
/// - "AKo" -> All offsuit AK combinations (12 combos)
/// - "AKs" -> All suited AK combinations (4 combos)
/// - "AK" -> All AK combinations (16 combos)
/// - "88" -> All pocket pairs (6 combos)
/// - "AhKs" -> Specific cards (1 combo)
pub fn parseNotation(notation: []const u8, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    if (notation.len < 2) return NotationError.InvalidLength;

    // Handle specific cards first (e.g., "AhKs")
    if (isSpecificCards(notation)) {
        const cards = try poker.parseCards(notation, allocator);
        defer allocator.free(cards);

        if (cards.len != 2) return NotationError.InvalidLength;

        var result = try allocator.alloc([2]poker.Card, 1);
        result[0] = [2]poker.Card{ cards[0], cards[1] };
        return result;
    }

    // Handle shorthand notation
    return parseShorthand(notation, allocator);
}

/// Check if notation represents specific cards (contains suit indicators like AhKs)
/// vs shorthand notation like AKs (suited) or AKo (offsuit)
fn isSpecificCards(notation: []const u8) bool {
    // Look for suit letters immediately after rank letters
    // Specific cards: "AhKs", "2c3d" etc.
    // Shorthand: "AKs", "AKo", "88" etc.

    var rank_count: u8 = 0;
    var suit_count: u8 = 0;

    for (notation) |c| {
        if (isRankChar(c)) {
            rank_count += 1;
        } else if (c == 'h' or c == 'd' or c == 'c') {
            suit_count += 1;
        } else if (c == 's') {
            // 's' could be a suit (spades) or shorthand indicator (suited)
            // If it's at the end and we have 2 ranks, it's shorthand
            // If it's in the middle/with other chars, it's likely a suit
            if (notation[notation.len - 1] == 's' and rank_count == 2 and suit_count == 0) {
                // This is shorthand like "AKs"
                return false;
            } else {
                // This is a suit like "As" or "AsBs"
                suit_count += 1;
            }
        }
    }

    // If we have suits (not just ranks), it's specific cards
    return suit_count > 0;
}

fn isRankChar(c: u8) bool {
    return switch (c) {
        '2', '3', '4', '5', '6', '7', '8', '9', 'T', 't', 'J', 'j', 'Q', 'q', 'K', 'k', 'A', 'a' => true,
        else => false,
    };
}

/// Parse shorthand notation like AKo, AKs, AK, 88
fn parseShorthand(notation: []const u8, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    // Handle pocket pairs (e.g., "AA", "KK", "88")
    if (notation.len == 2 and notation[0] == notation[1]) {
        return try generatePocketPair(notation[0], allocator);
    }

    // Handle two different ranks
    if (notation.len >= 2) {
        const rank1 = try parseRankChar(notation[0]);
        const rank2 = try parseRankChar(notation[1]);

        if (rank1 == rank2) {
            // This should be a pocket pair but wasn't caught above
            return try generatePocketPair(notation[0], allocator);
        }

        // Check for suited/offsuit indicator
        if (notation.len == 2) {
            // No indicator - return all combinations
            return try generateAllCombinations(rank1, rank2, allocator);
        } else if (notation.len == 3) {
            const indicator = notation[2];
            switch (indicator) {
                's' => return try generateSuitedCombinations(rank1, rank2, allocator),
                'o' => return try generateOffsuitCombinations(rank1, rank2, allocator),
                else => return NotationError.InvalidNotation,
            }
        }
    }

    return NotationError.InvalidNotation;
}

/// Parse a rank character to Rank enum
fn parseRankChar(c: u8) !poker.Rank {
    return switch (c) {
        '2' => poker.Rank.two,
        '3' => poker.Rank.three,
        '4' => poker.Rank.four,
        '5' => poker.Rank.five,
        '6' => poker.Rank.six,
        '7' => poker.Rank.seven,
        '8' => poker.Rank.eight,
        '9' => poker.Rank.nine,
        'T', 't' => poker.Rank.ten,
        'J', 'j' => poker.Rank.jack,
        'Q', 'q' => poker.Rank.queen,
        'K', 'k' => poker.Rank.king,
        'A', 'a' => poker.Rank.ace,
        else => NotationError.InvalidRank,
    };
}

/// Generate all combinations for a pocket pair (6 combinations)
fn generatePocketPair(rank_char: u8, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    const rank = try parseRankChar(rank_char);
    var combinations = try allocator.alloc([2]poker.Card, 6);

    const suits = [_]poker.Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations for pocket pairs
    for (suits, 0..) |suit1, i| {
        for (suits[i + 1 ..]) |suit2| {
            combinations[idx] = [2]poker.Card{
                poker.createCard(suit1, rank),
                poker.createCard(suit2, rank),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Generate all suited combinations (4 combinations)
fn generateSuitedCombinations(rank1: poker.Rank, rank2: poker.Rank, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    var combinations = try allocator.alloc([2]poker.Card, 4);
    const suits = [_]poker.Suit{ .hearts, .spades, .diamonds, .clubs };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]poker.Card{
            poker.createCard(suit, rank1),
            poker.createCard(suit, rank2),
        };
    }

    return combinations;
}

/// Generate all offsuit combinations (12 combinations)
fn generateOffsuitCombinations(rank1: poker.Rank, rank2: poker.Rank, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    var combinations = try allocator.alloc([2]poker.Card, 12);
    const suits = [_]poker.Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations where suits are different
    for (suits) |suit1| {
        for (suits) |suit2| {
            if (suit1 != suit2) {
                combinations[idx] = [2]poker.Card{
                    poker.createCard(suit1, rank1),
                    poker.createCard(suit2, rank2),
                };
                idx += 1;
            }
        }
    }

    return combinations;
}

/// Generate all combinations (suited + offsuit = 16 combinations)
fn generateAllCombinations(rank1: poker.Rank, rank2: poker.Rank, allocator: std.mem.Allocator) ![]const [2]poker.Card {
    var combinations = try allocator.alloc([2]poker.Card, 16);
    const suits = [_]poker.Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all possible suit combinations
    for (suits) |suit1| {
        for (suits) |suit2| {
            combinations[idx] = [2]poker.Card{
                poker.createCard(suit1, rank1),
                poker.createCard(suit2, rank2),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Get a single random combination from notation (for hand vs hand equity)
pub fn getRandomCombination(notation: []const u8, rng: std.Random, allocator: std.mem.Allocator) !?[2]poker.Card {
    const combinations = parseNotation(notation, allocator) catch |err| switch (err) {
        NotationError.InvalidNotation, NotationError.InvalidRank, NotationError.InvalidLength, NotationError.UnsupportedNotation => return null,
        else => return err,
    };
    defer allocator.free(combinations);

    if (combinations.len == 0) return null;

    const idx = rng.intRangeLessThan(usize, 0, combinations.len);
    return combinations[idx];
}

/// Count combinations for a notation
pub fn countCombinations(notation: []const u8) !u32 {
    if (isSpecificCards(notation)) return 1;

    if (notation.len == 2 and notation[0] == notation[1]) {
        return 6; // Pocket pair
    }

    if (notation.len >= 2) {
        if (notation.len == 2) {
            return 16; // All combinations
        } else if (notation.len == 3) {
            const indicator = notation[2];
            switch (indicator) {
                's' => return 4, // Suited
                'o' => return 12, // Offsuit
                else => return NotationError.InvalidNotation,
            }
        }
    }

    return NotationError.InvalidNotation;
}

// Tests
test "pocket pair parsing" {
    const allocator = std.testing.allocator;

    const combos = try parseNotation("AA", allocator);
    defer allocator.free(combos);

    try std.testing.expect(combos.len == 6);
}

test "suited combinations" {
    const allocator = std.testing.allocator;

    const combos = try parseNotation("AKs", allocator);
    defer allocator.free(combos);

    try std.testing.expect(combos.len == 4);
}

test "offsuit combinations" {
    const allocator = std.testing.allocator;

    const combos = try parseNotation("AKo", allocator);
    defer allocator.free(combos);

    try std.testing.expect(combos.len == 12);
}

test "all combinations" {
    const allocator = std.testing.allocator;

    const combos = try parseNotation("AK", allocator);
    defer allocator.free(combos);

    try std.testing.expect(combos.len == 16);
}

test "specific cards" {
    const allocator = std.testing.allocator;

    const combos = try parseNotation("AhKs", allocator);
    defer allocator.free(combos);

    try std.testing.expect(combos.len == 1);
}

test "combination counting" {
    try std.testing.expect(try countCombinations("AA") == 6);
    try std.testing.expect(try countCombinations("AKs") == 4);
    try std.testing.expect(try countCombinations("AKo") == 12);
    try std.testing.expect(try countCombinations("AK") == 16);
    try std.testing.expect(try countCombinations("AhKs") == 1);
}
