const std = @import("std");
const card = @import("card");
const hand = @import("hand");
const equity = @import("equity");

/// Range representation and poker notation parsing
/// Handles range strings like "AA,KK,AKs,AKo" and converts them to hand combinations
/// Notation parsing is implemented inline for efficient direct insertion into range HashMap

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
        if (notation.len < 2 or notation.len > 3) return error.InvalidNotation;

        const rank1 = parseRank(notation[0]) orelse return error.InvalidRank;
        const rank2 = parseRank(notation[1]) orelse return error.InvalidRank;
        const rank1_idx = @as(u8, @intCast(rank1 - 2)); // Convert poker rank to card rank (0-12)
        const rank2_idx = @as(u8, @intCast(rank2 - 2));

        if (notation.len == 2) {
            if (rank1 == rank2) {
                // Pocket pair - add all 6 combinations
                const combinations = try hand.generatePocketPair(@enumFromInt(rank1_idx), self.allocator);
                defer self.allocator.free(combinations);

                for (combinations) |combo| {
                    try self.addHand(combo, probability);
                }
            } else {
                // Unpaired hand without modifier - add all 16 combinations (suited + offsuit)
                const combinations = try hand.generateAllCombinations(
                    @enumFromInt(rank1_idx),
                    @enumFromInt(rank2_idx),
                    self.allocator,
                );
                defer self.allocator.free(combinations);

                for (combinations) |combo| {
                    try self.addHand(combo, probability);
                }
            }
        } else {
            // Suited or offsuit (e.g., "AKs", "AKo")
            const modifier = notation[2];
            if (rank1 == rank2) return error.CannotBeSuited;

            switch (modifier) {
                's', 'S' => {
                    // Suited - add 4 combinations
                    const combinations = try hand.generateSuitedCombinations(
                        @enumFromInt(rank1_idx),
                        @enumFromInt(rank2_idx),
                        self.allocator,
                    );
                    defer self.allocator.free(combinations);

                    for (combinations) |combo| {
                        try self.addHand(combo, probability);
                    }
                },
                'o', 'O' => {
                    // Offsuit - add 12 combinations
                    const combinations = try hand.generateOffsuitCombinations(
                        @enumFromInt(rank1_idx),
                        @enumFromInt(rank2_idx),
                        self.allocator,
                    );
                    defer self.allocator.free(combinations);

                    for (combinations) |combo| {
                        try self.addHand(combo, probability);
                    }
                },
                else => return error.InvalidModifier,
            }
        }
    }

    // Note: parseRank helper function is defined below

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

/// Parse single rank character to numeric value (case-insensitive)
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
        'T', 't' => 10,
        'J', 'j' => 11,
        'Q', 'q' => 12,
        'K', 'k' => 13,
        'A', 'a' => 14,
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

            // Calculate exact equity for this hand combination
            const hero_combined = hero_entry.hand[0] | hero_entry.hand[1];
            const villain_combined = villain_entry.hand[0] | villain_entry.hand[1];
            const equity_result = try equity.exact(
                hero_combined,
                villain_combined,
                board,
                allocator,
            );
            const hero_win_rate = equity_result.equity();

            // Weight by probabilities of both hands
            const weight = hero_entry.probability * villain_entry.probability;
            total_hero_equity += hero_win_rate * weight;
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

/// Sample a hand from a range with weighted probability
fn sampleHandFromRange(range: *const Range, total_weight: f64, rng: std.Random) [2]Hand {
    const target = rng.float(f64) * total_weight;
    var cumulative: f64 = 0;

    var iter = range.iterator();
    while (iter.next()) |entry| {
        cumulative += entry.probability;
        if (cumulative >= target) {
            return entry.hand;
        }
    }

    // Fallback - return first hand found (should not happen with proper weights)
    var fallback_iter = range.iterator();
    if (fallback_iter.next()) |entry| {
        return entry.hand;
    }
    // This should never happen
    unreachable;
}

/// Calculate range vs range equity using Monte Carlo simulation
/// Note: This function will need to import equity module once it's created
pub fn calculateRangeEquityMonteCarlo(hero_range: *const Range, villain_range: *const Range, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !RangeEquityResult {
    // Sample hands from ranges and run Monte Carlo simulations
    var hero_wins: u32 = 0;
    var ties: u32 = 0;

    // Pre-calculate total weights for weighted sampling
    var hero_total_weight: f64 = 0;
    var hero_iter = hero_range.iterator();
    while (hero_iter.next()) |entry| {
        hero_total_weight += entry.probability;
    }

    var villain_total_weight: f64 = 0;
    var villain_iter = villain_range.iterator();
    while (villain_iter.next()) |entry| {
        villain_total_weight += entry.probability;
    }

    var i: u32 = 0;
    while (i < simulations) : (i += 1) {
        // Sample hero hand from range
        const hero_hand = sampleHandFromRange(hero_range, hero_total_weight, rng);
        const villain_hand = sampleHandFromRange(villain_range, villain_total_weight, rng);

        // Skip if hands conflict
        if (hasCardConflict(hero_hand, villain_hand, board)) {
            continue;
        }

        // Run Monte Carlo simulation for this matchup
        const hero_combined = hero_hand[0] | hero_hand[1];
        const villain_combined = villain_hand[0] | villain_hand[1];
        const result = try equity.monteCarlo(
            hero_combined,
            villain_combined,
            board,
            1, // Single simulation since we're already sampling
            rng,
            allocator,
        );

        if (result.wins > 0) {
            hero_wins += 1;
        } else if (result.ties > 0) {
            ties += 1;
        }
    }

    const total = hero_wins + ties + (simulations - hero_wins - ties);
    const hero_equity = (@as(f64, @floatFromInt(hero_wins)) + @as(f64, @floatFromInt(ties)) * 0.5) / @as(f64, @floatFromInt(total));

    return RangeEquityResult{
        .hero_equity = hero_equity,
        .villain_equity = 1.0 - hero_equity,
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
        card.makeCard(.clubs, .ace), // Ace of clubs
        card.makeCard(.diamonds, .ace), // Ace of diamonds
    };
    try range.addHand(hole_hand, 1.0);

    try testing.expect(range.handCount() == 1);
    try testing.expect(range.getHandProbability(hole_hand) == 1.0);
}

test "hand key conversion" {
    const hand1 = [2]Hand{
        card.makeCard(.hearts, .ace), // Ace of hearts
        card.makeCard(.spades, .ace), // Ace of spades
    };

    const hand_key = HandKey.init(hand1);
    const reconstructed = hand_key.toHand();

    // Check that the reconstructed bits match the original bits
    const original_bits = hand1[0] | hand1[1];
    const reconstructed_bits = reconstructed[0] | reconstructed[1];
    try testing.expect(original_bits == reconstructed_bits);
}

test "range notation parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test parseRank function
    try testing.expectEqual(@as(?u8, 2), parseRank('2'));
    try testing.expectEqual(@as(?u8, 10), parseRank('T'));
    try testing.expectEqual(@as(?u8, 14), parseRank('A'));
    try testing.expectEqual(@as(?u8, null), parseRank('X'));

    // Test range with different notations
    var range = Range.init(allocator);
    defer range.deinit();

    // Add pocket pairs
    try range.addHandNotation("AA", 1.0);
    try testing.expect(range.handCount() == 6); // 6 combinations for AA

    // Add suited hand
    try range.addHandNotation("AKs", 1.0);
    try testing.expect(range.handCount() == 10); // 6 + 4 = 10

    // Add offsuit hand
    try range.addHandNotation("QJo", 1.0);
    try testing.expect(range.handCount() == 22); // 6 + 4 + 12 = 22
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
    // Test that we properly use hand.parseHand (CardSet approach)
    const single_card = hand.parseHand("As");
    try testing.expect(card.countCards(single_card) == 1);
    try testing.expect(card.hasCard(single_card, .spades, .ace));

    const hole_cards = hand.parseHand("AhAs");
    try testing.expect(card.countCards(hole_cards) == 2);
    try testing.expect(card.hasCard(hole_cards, .hearts, .ace));
    try testing.expect(card.hasCard(hole_cards, .spades, .ace));
}

test "case-insensitive notation parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test lowercase ranks
    var range1 = try parseRange("aa,kk,qq", allocator);
    defer range1.deinit();
    try testing.expect(range1.handCount() == 18); // 6 + 6 + 6

    // Test mixed case
    var range2 = try parseRange("Aa,kK,AkS,aKo", allocator);
    defer range2.deinit();
    try testing.expect(range2.handCount() == 28); // 6 + 6 + 4 + 12

    // Test specific case patterns
    var range3 = Range.init(allocator);
    defer range3.deinit();
    try range3.addHandNotation("jj", 1.0);
    try testing.expect(range3.handCount() == 6);

    try range3.addHandNotation("AkS", 1.0);
    try testing.expect(range3.handCount() == 10); // 6 + 4

    try range3.addHandNotation("qJo", 1.0);
    try testing.expect(range3.handCount() == 22); // 6 + 4 + 12
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
