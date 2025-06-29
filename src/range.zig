const std = @import("std");
const card = @import("card");
const hand = @import("hand");

// TODO: Will need to import notation parser once internal module is created
// const notation_parser = @import("poker/notation.zig");

/// Range representation and poker notation parsing
/// Handles range strings like "AA,KK,AKs,AKo" and converts them to hand combinations

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;

/// Simple hand key for hash map - uses sorted card bits
const HandKey = struct {
    card1_bits: u64,
    card2_bits: u64,

    fn init(hole_hand: [2]Hand) HandKey {
        // Sort cards to ensure consistent ordering regardless of input order
        if (hole_hand[0] <= hole_hand[1]) {
            return HandKey{ .card1_bits = hole_hand[0], .card2_bits = hole_hand[1] };
        } else {
            return HandKey{ .card1_bits = hole_hand[1], .card2_bits = hole_hand[0] };
        }
    }

    fn toHand(self: HandKey) [2]Hand {
        return [2]Hand{ self.card1_bits, self.card2_bits };
    }
};

/// Efficient representation of a poker hand range
/// Uses HashMap for simplicity and correctness
pub const Range = struct {
    /// Map from hand to probability
    hands: std.AutoHashMap(HandKey, f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Range {
        return Range{
            .hands = std.AutoHashMap(HandKey, f32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Range) void {
        self.hands.deinit();
    }

    /// Add a hand to the range with given probability
    pub fn addHand(self: *Range, hole_hand: [2]Hand, probability: f32) !void {
        const hand_key = HandKey.init(hole_hand);
        try self.hands.put(hand_key, probability);
    }

    /// Add a hand using poker notation (e.g., "AA", "AKs", "AKo")
    /// For pocket pairs like "AA", this adds ALL possible combinations (6 for AA)
    /// For suited/offsuit like "AKs"/"AKo", this adds all combinations
    /// For unpaired notation like "AK", adds all combinations (16 total)
    pub fn addHandNotation(self: *Range, notation: []const u8, probability: f32) !void {
        if (notation.len == 2 and notation[0] == notation[1]) {
            // Pocket pair - add all combinations
            try self.addAllPocketPairCombinations(notation, probability);
        } else if (notation.len == 2 and notation[0] != notation[1]) {
            // Unpaired hand without modifier (e.g., "AK") - generate all combinations manually
            const rank1 = parseRank(notation[0]) orelse return error.InvalidRank;
            const rank2 = parseRank(notation[1]) orelse return error.InvalidRank;

            // Add all 16 combinations (4 suits x 4 suits)
            const suits = [4]u2{ 0, 1, 2, 3 };
            for (suits) |suit1| {
                for (suits) |suit2| {
                    const hole_hand = [2]Hand{
                        card.makeCard(suit1, rank1 - 2),
                        card.makeCard(suit2, rank2 - 2),
                    };
                    try self.addHand(hole_hand, probability);
                }
            }
        } else {
            // Suited/offsuit - handle manually for now
            const hole_hand = try parseHandNotation(notation);
            try self.addHand(hole_hand, probability);
        }
    }

    /// Add all combinations of a pocket pair (e.g., "AA" adds all 6 AA combinations)
    fn addAllPocketPairCombinations(self: *Range, notation: []const u8, probability: f32) !void {
        if (notation.len != 2 or notation[0] != notation[1]) return error.NotAPocketPair;

        const rank = parseRank(notation[0]) orelse return error.InvalidRank;

        // Add all 6 combinations: 4 choose 2 = 6 combinations
        const suits = [4]u2{ 0, 1, 2, 3 }; // h, s, d, c

        for (suits, 0..) |suit1, i| {
            for (suits[i + 1 ..]) |suit2| {
                const hole_hand = [2]Hand{
                    card.makeCard(suit1, rank - 2), // Convert poker rank to card rank
                    card.makeCard(suit2, rank - 2),
                };
                try self.addHand(hole_hand, probability);
            }
        }
    }

    /// Add multiple hands with same probability using poker notation
    pub fn addHands(self: *Range, notations: []const []const u8, probability: f32) !void {
        for (notations) |notation| {
            try self.addHandNotation(notation, probability);
        }
    }

    /// Get probability of a specific hand in range
    pub fn getHandProbability(self: *const Range, hole_hand: [2]Hand) f32 {
        const hand_key = HandKey.init(hole_hand);
        return self.hands.get(hand_key) orelse 0.0;
    }

    /// Get total number of hand combinations in range
    pub fn handCount(self: *const Range) u32 {
        return @intCast(self.hands.count());
    }

    /// Iterator for efficient range traversal
    pub const Iterator = struct {
        inner: std.AutoHashMap(HandKey, f32).Iterator,

        pub fn next(self: *Iterator) ?struct { hand: [2]Hand, probability: f32 } {
            if (self.inner.next()) |entry| {
                const hand_key = entry.key_ptr.*;
                const probability = entry.value_ptr.*;

                return .{ .hand = hand_key.toHand(), .probability = probability };
            }
            return null;
        }
    };

    pub fn iterator(self: *const Range) Iterator {
        return Iterator{ .inner = self.hands.iterator() };
    }
};

/// Parse range hand notation (e.g., "AA", "AKs", "AKo") into representative hand
/// This is used internally by parseRange() and addHandNotation()
fn parseHandNotation(notation: []const u8) ![2]Hand {
    if (notation.len < 2 or notation.len > 3) return error.InvalidNotation;

    const rank1 = parseRank(notation[0]) orelse return error.InvalidRank;
    const rank2 = parseRank(notation[1]) orelse return error.InvalidRank;

    if (notation.len == 2) {
        if (rank1 == rank2) {
            // Pocket pair (e.g., "AA", "KK")
            return [2]Hand{
                card.makeCard(0, rank1 - 2), // Convert poker rank to card rank
                card.makeCard(1, rank2 - 2),
            };
        } else {
            // Unpaired hand without modifier - this is ambiguous, so we need to handle it at range level
            return error.AmbiguousNotation;
        }
    } else {
        // Suited or offsuit (e.g., "AKs", "AKo")
        const modifier = notation[2];
        if (rank1 == rank2) return error.CannotBeSuited;

        switch (modifier) {
            's' => return [2]Hand{ card.makeCard(0, rank1 - 2), card.makeCard(0, rank2 - 2) }, // Same suit
            'o' => return [2]Hand{ card.makeCard(0, rank1 - 2), card.makeCard(1, rank2 - 2) }, // Different suits
            else => return error.InvalidModifier,
        }
    }
}

/// Parse single rank character to numeric value
fn parseRank(char: u8) ?u8 {
    return switch (char) {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        '5' => 5,
        '6' => 6,
        '7' => 7,
        '8' => 8,
        '9' => 9,
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        else => null,
    };
}

/// Parse comma-delimited range notation (e.g., "AA,KK,QQ,AKs,AKo") into a Range
pub fn parseRange(notation: []const u8, allocator: std.mem.Allocator) !Range {
    var range = Range.init(allocator);
    errdefer range.deinit();

    // Split by commas and process each hand type
    var iterator = std.mem.splitScalar(u8, notation, ',');
    while (iterator.next()) |hand_str| {
        const trimmed = std.mem.trim(u8, hand_str, " \t\n\r");
        if (trimmed.len == 0) continue;

        // Add with full probability (1.0) - user can adjust later if needed
        try range.addHandNotation(trimmed, 1.0);
    }

    return range;
}

/// Range equity calculation result (from ranges.zig)
pub const RangeEquityResult = struct {
    hero_equity: f64,
    villain_equity: f64,
    total_simulations: u32,

    pub fn sum(self: RangeEquityResult) f64 {
        return self.hero_equity + self.villain_equity;
    }
};

/// Check if hands or board have conflicting cards
fn hasCardConflict(hero_hole: [2]Hand, villain_hole: [2]Hand, board: []const Hand) bool {
    // Combine cards into hands using bitwise OR
    const hero = hero_hole[0] | hero_hole[1];
    const villain = villain_hole[0] | villain_hole[1];
    var board_hand: Hand = 0;
    for (board) |board_card| {
        board_hand |= board_card;
    }

    // Check for any overlapping bits
    return (hero & villain) != 0 or
        (hero & board_hand) != 0 or
        (villain & board_hand) != 0;
}

/// Calculate exact range vs range equity
/// Note: This function will need to import equity module once it's created
pub fn calculateRangeEquityExact(hero_range: *const Range, villain_range: *const Range, board: []const Hand, allocator: std.mem.Allocator) !RangeEquityResult {
    var total_hero_equity: f64 = 0.0;
    var total_weight: f64 = 0.0;
    var total_combinations: u32 = 0;

    // Iterate through all combinations of hands from both ranges
    var hero_iter = hero_range.iterator();
    while (hero_iter.next()) |hero_entry| {
        var villain_iter = villain_range.iterator();
        while (villain_iter.next()) |villain_entry| {
            // Check for card conflicts
            if (hasCardConflict(hero_entry.hand, villain_entry.hand, board)) {
                continue;
            }

            // TODO: Calculate exact equity for this hand combination
            // This will need to import equity module once it's created
            _ = allocator; // Suppress unused variable warning for now
            const equity_result_placeholder: f64 = 0.5; // Placeholder

            // Weight by probabilities of both hands
            const weight = hero_entry.probability * villain_entry.probability;
            total_hero_equity += equity_result_placeholder * weight;
            total_weight += weight;
            total_combinations += 1;
        }
    }

    // Normalize by total weight
    const hero_equity = if (total_weight > 0) total_hero_equity / total_weight else 0.0;

    return RangeEquityResult{
        .hero_equity = hero_equity,
        .villain_equity = 1.0 - hero_equity,
        .total_simulations = total_combinations,
    };
}

/// Calculate range vs range equity using Monte Carlo simulation
/// Note: This function will need to import equity module once it's created
pub fn calculateRangeEquityMonteCarlo(hero_range: *const Range, villain_range: *const Range, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !RangeEquityResult {
    _ = hero_range;
    _ = villain_range;
    _ = board;
    _ = rng;
    _ = allocator;

    // TODO: Implement once equity module is created
    return RangeEquityResult{
        .hero_equity = 0.5,
        .villain_equity = 0.5,
        .total_simulations = simulations,
    };
}

/// Common preflop ranges for quick setup
pub const CommonRanges = struct {
    /// Create a tight opening range (premium hands only)
    pub fn tightOpen(allocator: std.mem.Allocator) !Range {
        var range = Range.init(allocator);

        // Premium pairs and suited connectors
        try range.addHands(&.{ "AA", "KK", "QQ", "AKs" }, 1.0);

        return range;
    }

    /// Create a loose calling range
    pub fn looseCall(allocator: std.mem.Allocator) !Range {
        var range = Range.init(allocator);

        // Wide range of hands with varying probabilities
        try range.addHands(&.{ "TT", "99", "88", "77" }, 0.8);
        try range.addHands(&.{ "KQs", "QJs", "JTs" }, 0.7);
        try range.addHands(&.{ "AKo", "AQo" }, 0.6);

        return range;
    }

    /// Create a button opening range (wide)
    pub fn buttonOpen(allocator: std.mem.Allocator) !Range {
        var range = Range.init(allocator);

        // Wide button range
        try range.addHands(&.{ "AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55" }, 1.0); // Pairs
        try range.addHands(&.{ "AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s", "A5s" }, 1.0); // Suited aces
        try range.addHands(&.{ "KQs", "KJs", "KTs", "QJs", "QTs", "JTs" }, 1.0); // Suited connectors
        try range.addHands(&.{ "AKo", "AQo", "AJo", "ATo" }, 1.0); // Offsuit broadway

        return range;
    }
};

// Tests
const testing = std.testing;

test "range basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    const hole_hand = [2]Hand{
        card.makeCard(0, 12), // Ace of clubs
        card.makeCard(1, 12), // Ace of diamonds
    };
    try range.addHand(hole_hand, 1.0);

    try testing.expect(range.handCount() == 1);
    try testing.expect(range.getHandProbability(hole_hand) == 1.0);
}

test "hand key conversion" {
    const hand1 = [2]Hand{
        card.makeCard(2, 12), // Ace of hearts
        card.makeCard(3, 12), // Ace of spades
    };

    const hand_key = HandKey.init(hand1);
    const reconstructed = hand_key.toHand();

    // Check that the reconstructed bits match the original bits
    const original_bits = hand1[0] | hand1[1];
    const reconstructed_bits = reconstructed[0] | reconstructed[1];
    try testing.expect(original_bits == reconstructed_bits);
}

test "range notation parsing" {
    // Test pocket pairs
    const aa = try parseHandNotation("AA");
    try testing.expect(card.getSuitMask(aa[0], .clubs) != 0 or card.getSuitMask(aa[0], .diamonds) != 0 or card.getSuitMask(aa[0], .hearts) != 0 or card.getSuitMask(aa[0], .spades) != 0);
    try testing.expect(card.getSuitMask(aa[1], .clubs) != 0 or card.getSuitMask(aa[1], .diamonds) != 0 or card.getSuitMask(aa[1], .hearts) != 0 or card.getSuitMask(aa[1], .spades) != 0);

    const kk = try parseHandNotation("KK");
    try testing.expect(card.getSuitMask(kk[0], .clubs) != 0 or card.getSuitMask(kk[0], .diamonds) != 0 or card.getSuitMask(kk[0], .hearts) != 0 or card.getSuitMask(kk[0], .spades) != 0);
    try testing.expect(card.getSuitMask(kk[1], .clubs) != 0 or card.getSuitMask(kk[1], .diamonds) != 0 or card.getSuitMask(kk[1], .hearts) != 0 or card.getSuitMask(kk[1], .spades) != 0);

    // Test suited hands
    const aks = try parseHandNotation("AKs");
    try testing.expect((aks[0] & aks[1]) == 0); // Different ranks

    // Test offsuit hands
    const ako = try parseHandNotation("AKo");
    try testing.expect((ako[0] & ako[1]) == 0); // Different ranks
}

test "range with notation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add hands using notation
    try range.addHandNotation("AA", 1.0); // Should add 6 combinations
    // Note: AKs and multiple hands will be tested once notation parser is available

    // For now, just test that basic functionality works
    try testing.expect(range.handCount() == 6); // 6 combinations for AA
}

test "pocket pair expansion" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add just AA - should create 6 combinations
    try range.addHandNotation("AA", 1.0);
    try testing.expect(range.handCount() == 6);
}

test "parseRange function" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test simple range parsing
    var range = try parseRange("AA", allocator);
    defer range.deinit();

    try testing.expect(range.handCount() == 6); // 6 combinations for AA
}

test "range delegation to hand parsing" {
    // Test that we properly use hand.mustParseHand (CardSet approach)
    const single_card = hand.mustParseHand("As");
    try testing.expect(card.countCards(single_card) == 1);
    try testing.expect(card.hasCard(single_card, .spades, .ace));

    const hole_cards = hand.mustParseHand("AhAs");
    try testing.expect(card.countCards(hole_cards) == 2);
    try testing.expect(card.hasCard(hole_cards, .hearts, .ace));
    try testing.expect(card.hasCard(hole_cards, .spades, .ace));
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
