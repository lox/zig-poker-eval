const std = @import("std");
const poker = @import("poker.zig");
const notation_parser = @import("notation.zig");
const simulation = @import("simulation.zig");

/// Simple hand key for hash map - uses sorted card bits
const HandKey = struct {
    card1_bits: u64,
    card2_bits: u64,

    fn init(hand: [2]poker.Card) HandKey {
        // Sort cards to ensure consistent ordering regardless of input order
        if (hand[0].bits <= hand[1].bits) {
            return HandKey{ .card1_bits = hand[0].bits, .card2_bits = hand[1].bits };
        } else {
            return HandKey{ .card1_bits = hand[1].bits, .card2_bits = hand[0].bits };
        }
    }

    fn toHand(self: HandKey) [2]poker.Card {
        return [2]poker.Card{ poker.Card{ .bits = self.card1_bits }, poker.Card{ .bits = self.card2_bits } };
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
    pub fn addHand(self: *Range, hand: [2]poker.Card, probability: f32) !void {
        const hand_key = HandKey.init(hand);
        try self.hands.put(hand_key, probability);
    }

    /// Add a hand using poker notation (e.g., "AA", "AKs", "AKo")
    /// For pocket pairs like "AA", this adds ALL possible combinations (6 for AA)
    /// For suited/offsuit like "AKs"/"AKo", this adds one representative hand
    /// For unpaired notation like "AK", adds all combinations (16 total)
    pub fn addHandNotation(self: *Range, notation: []const u8, probability: f32) !void {
        if (notation.len == 2 and notation[0] == notation[1]) {
            // Pocket pair - add all combinations
            try self.addAllPocketPairCombinations(notation, probability);
        } else if (notation.len == 2 and notation[0] != notation[1]) {
            // Unpaired hand without modifier (e.g., "AK") - use notation parser for all combinations
            const allocator = std.heap.page_allocator;
            const combinations = notation_parser.parse(notation, allocator) catch {
                // Fallback to single representative hand if parsing fails
                const hand = try parseHandNotation(notation);
                try self.addHand(hand, probability);
                return;
            };
            defer allocator.free(combinations);

            for (combinations) |combo| {
                try self.addHand(combo, probability);
            }
        } else {
            // Suited/offsuit - use notation parser for all combinations
            const allocator = std.heap.page_allocator;
            const combinations = notation_parser.parse(notation, allocator) catch {
                // Fallback to single representative hand if parsing fails
                const hand = try parseHandNotation(notation);
                try self.addHand(hand, probability);
                return;
            };
            defer allocator.free(combinations);

            for (combinations) |combo| {
                try self.addHand(combo, probability);
            }
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
                const hand = [2]poker.Card{
                    poker.Card.init(rank, suit1),
                    poker.Card.init(rank, suit2),
                };
                try self.addHand(hand, probability);
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
    pub fn getHandProbability(self: *const Range, hand: [2]poker.Card) f32 {
        const hand_key = HandKey.init(hand);
        return self.hands.get(hand_key) orelse 0.0;
    }

    /// Get total number of hand combinations in range
    pub fn handCount(self: *const Range) u32 {
        return @intCast(self.hands.count());
    }

    /// Iterator for efficient range traversal
    pub const Iterator = struct {
        inner: std.AutoHashMap(HandKey, f32).Iterator,

        pub fn next(self: *Iterator) ?struct { hand: [2]poker.Card, probability: f32 } {
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
fn parseHandNotation(notation: []const u8) ![2]poker.Card {
    if (notation.len < 2 or notation.len > 3) return error.InvalidNotation;

    const rank1 = parseRank(notation[0]) orelse return error.InvalidRank;
    const rank2 = parseRank(notation[1]) orelse return error.InvalidRank;

    if (notation.len == 2) {
        if (rank1 == rank2) {
            // Pocket pair (e.g., "AA", "KK")
            return [2]poker.Card{ poker.Card.init(rank1, 0), poker.Card.init(rank2, 1) };
        } else {
            // Unpaired hand without modifier - this is ambiguous, so we need to handle it at range level
            return error.AmbiguousNotation;
        }
    } else {
        // Suited or offsuit (e.g., "AKs", "AKo")
        const modifier = notation[2];
        if (rank1 == rank2) return error.CannotBeSuited;

        switch (modifier) {
            's' => return [2]poker.Card{ poker.Card.init(rank1, 0), poker.Card.init(rank2, 0) }, // Same suit
            'o' => return [2]poker.Card{ poker.Card.init(rank1, 0), poker.Card.init(rank2, 1) }, // Different suits
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

/// Range equity calculation result
pub const RangeEquityResult = struct {
    hero_equity: f64,
    villain_equity: f64,
    total_simulations: u32,

    pub fn sum(self: RangeEquityResult) f64 {
        return self.hero_equity + self.villain_equity;
    }
};

/// Calculate exact range vs range equity
pub fn calculateRangeEquityExact(hero_range: *const Range, villain_range: *const Range, board: []const poker.Card, allocator: std.mem.Allocator) !RangeEquityResult {
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
            const equity = @import("equity.zig");
            const equity_result = try equity.exact(hero_entry.hand, villain_entry.hand, board, allocator);

            // Weight by probabilities of both hands
            const weight = hero_entry.probability * villain_entry.probability;
            total_hero_equity += equity_result.equity() * weight;
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
pub fn calculateRangeEquityMonteCarlo(hero_range: *const Range, villain_range: *const Range, board: []const poker.Card, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !RangeEquityResult {
    var total_hero_equity: f64 = 0.0;
    var simulation_count: u32 = 0;

    for (0..simulations) |_| {
        // Sample hands from ranges based on probabilities
        const hero_hand = sampleHandFromRange(hero_range, rng) orelse continue;
        const villain_hand = sampleHandFromRange(villain_range, rng) orelse continue;

        // Check for card conflicts
        if (hasCardConflict(hero_hand, villain_hand, board)) {
            continue;
        }

        // Calculate equity for sampled hands
        const equity = @import("equity.zig");
        const equity_result = try equity.monteCarlo(hero_hand, villain_hand, board, 100, // Sub-simulations per range sample
            rng, allocator);

        total_hero_equity += equity_result.equity();
        simulation_count += 1;
    }

    const hero_equity = if (simulation_count > 0) total_hero_equity / @as(f64, @floatFromInt(simulation_count)) else 0.0;

    return RangeEquityResult{
        .hero_equity = hero_equity,
        .villain_equity = 1.0 - hero_equity,
        .total_simulations = simulation_count,
    };
}

/// Sample a hand from a range based on probabilities
fn sampleHandFromRange(range: *const Range, rng: std.Random) ?[2]poker.Card {
    if (range.handCount() == 0) return null;

    // Calculate total probability first to normalize
    var total_prob: f32 = 0.0;
    var iter = range.iterator();
    while (iter.next()) |entry| {
        total_prob += entry.probability;
    }

    if (total_prob <= 0.0) return null;

    // Generate random value and sample proportionally
    const random_value = rng.float(f32) * total_prob;
    var cumulative_prob: f32 = 0.0;
    var last_hand: ?[2]poker.Card = null;

    var sample_iter = range.iterator();
    while (sample_iter.next()) |entry| {
        last_hand = entry.hand; // Keep track of last hand for floating-point edge cases
        cumulative_prob += entry.probability;
        if (random_value <= cumulative_prob) {
            return entry.hand;
        }
    }

    return last_hand; // Fallback for floating-point precision edge cases
}

/// Check if hands or board have conflicting cards
fn hasCardConflict(hero_hand: [2]poker.Card, villain_hand: [2]poker.Card, board: []const poker.Card) bool {
    // Use Hand objects for cleaner bit operations
    const hero = poker.Hand{ .bits = hero_hand[0].bits | hero_hand[1].bits };
    const villain = poker.Hand{ .bits = villain_hand[0].bits | villain_hand[1].bits };
    const board_hand = poker.Hand.fromBoard(board);

    // Check for any overlapping bits using Hand methods
    return hero.hasConflictWith(villain) or
        hero.hasConflictWith(board_hand) or
        villain.hasConflictWith(board_hand);
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
test "range basic operations" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    const hand = [2]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AA
    try range.addHand(hand, 1.0);

    try std.testing.expect(range.handCount() == 1);
    try std.testing.expect(range.getHandProbability(hand) == 1.0);
}

test "hand key conversion" {
    const hand1 = [2]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // Ah As

    const hand_key = HandKey.init(hand1);
    const reconstructed = hand_key.toHand();

    // Check that the reconstructed bits match the original bits
    const original_bits = hand1[0].bits | hand1[1].bits;
    const reconstructed_bits = reconstructed[0].bits | reconstructed[1].bits;
    try std.testing.expect(original_bits == reconstructed_bits);
}

test "range notation parsing" {
    // Test pocket pairs
    const aa = try parseHandNotation("AA");
    try std.testing.expect(aa[0].getRank() == 14 and aa[1].getRank() == 14);
    try std.testing.expect(aa[0].getSuit() != aa[1].getSuit()); // Different suits for pairs

    const kk = try parseHandNotation("KK");
    try std.testing.expect(kk[0].getRank() == 13 and kk[1].getRank() == 13);

    // Test suited hands
    const aks = try parseHandNotation("AKs");
    try std.testing.expect(aks[0].getRank() == 14 and aks[1].getRank() == 13);
    try std.testing.expect(aks[0].getSuit() == aks[1].getSuit()); // Same suit

    // Test offsuit hands
    const ako = try parseHandNotation("AKo");
    try std.testing.expect(ako[0].getRank() == 14 and ako[1].getRank() == 13);
    try std.testing.expect(ako[0].getSuit() != ako[1].getSuit()); // Different suits
}

test "range with notation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add hands using notation
    try range.addHandNotation("AA", 1.0); // Should add 6 combinations
    try range.addHandNotation("AKs", 0.8); // Should add 1 combination

    // Add multiple pocket pairs
    try range.addHands(&.{ "KK", "QQ" }, 1.0); // Should add 6 + 6 = 12 combinations

    // Total: 6 (AA) + 4 (AKs) + 6 (KK) + 6 (QQ) = 22 combinations
    try std.testing.expect(range.handCount() == 22);
}

test "pocket pair expansion" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add just AA - should create 6 combinations
    try range.addHandNotation("AA", 1.0);
    try std.testing.expect(range.handCount() == 6);

    // Verify difference between range and specific hand parsing
    // parseCards returns specific cards vs range which expands to all combinations
}

test "parseRange function" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test comma-delimited range parsing
    var range = try parseRange("AA,KK,AKs", allocator);
    defer range.deinit();

    // Should have: 6 (AA) + 6 (KK) + 4 (AKs all combinations) = 16 combinations
    try std.testing.expect(range.handCount() == 16);
}

// Table-driven tests for range notation parsing
test "range notation parsing - comprehensive" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const test_cases = [_]struct {
        notation: []const u8,
        expected_count: u32,
        should_work: bool,
        description: []const u8,
    }{
        // Pocket pairs
        .{ .notation = "AA", .expected_count = 6, .should_work = true, .description = "Pocket aces (6 combinations)" },
        .{ .notation = "KK", .expected_count = 6, .should_work = true, .description = "Pocket kings" },
        .{ .notation = "22", .expected_count = 6, .should_work = true, .description = "Pocket deuces" },

        // Suited hands
        .{ .notation = "AKs", .expected_count = 4, .should_work = true, .description = "Ace-king suited" },
        .{ .notation = "AQs", .expected_count = 4, .should_work = true, .description = "Ace-queen suited" },
        .{ .notation = "KQs", .expected_count = 4, .should_work = true, .description = "King-queen suited" },

        // Offsuit hands
        .{ .notation = "AKo", .expected_count = 12, .should_work = true, .description = "Ace-king offsuit" },
        .{ .notation = "AQo", .expected_count = 12, .should_work = true, .description = "Ace-queen offsuit" },
        .{ .notation = "KQo", .expected_count = 12, .should_work = true, .description = "King-queen offsuit" },

        // Any hands (both suited and offsuit)
        .{ .notation = "AK", .expected_count = 16, .should_work = true, .description = "Ace-king any (suited + offsuit)" },
        .{ .notation = "AQ", .expected_count = 16, .should_work = true, .description = "Ace-queen any" },
        .{ .notation = "KQ", .expected_count = 16, .should_work = true, .description = "King-queen any" },

        // Invalid cases
        .{ .notation = "AKx", .expected_count = 0, .should_work = false, .description = "Invalid modifier" },
        .{ .notation = "XY", .expected_count = 0, .should_work = false, .description = "Invalid ranks" },
        .{ .notation = "A", .expected_count = 0, .should_work = false, .description = "Too short" },
        .{ .notation = "AKQJ", .expected_count = 0, .should_work = false, .description = "Too long" },
    };

    for (test_cases) |test_case| {
        var range = Range.init(allocator);
        defer range.deinit();

        const result = range.addHandNotation(test_case.notation, 1.0);

        if (test_case.should_work) {
            result catch |err| {
                std.debug.print("FAIL: {s} - Expected to work but got error: {}\n", .{ test_case.description, err });
                try std.testing.expect(false);
            };

            const actual_count = range.handCount();
            if (actual_count != test_case.expected_count) {
                std.debug.print("FAIL: {s} - Expected {} hands, got {}\n", .{ test_case.description, test_case.expected_count, actual_count });
            }
            try std.testing.expectEqual(test_case.expected_count, actual_count);
        } else {
            if (result) |_| {
                std.debug.print("FAIL: {s} - Expected to fail but succeeded\n", .{test_case.description});
                try std.testing.expect(false);
            } else |_| {
                // Test passed - expected failure occurred
            }
        }
    }
}

test "range notation parsing - comma separated ranges" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const test_cases = [_]struct {
        range_str: []const u8,
        expected_count: u32,
        description: []const u8,
    }{
        .{ .range_str = "AA,KK,QQ", .expected_count = 18, .description = "Three pocket pairs (6+6+6)" },
        .{ .range_str = "AKs,AKo", .expected_count = 16, .description = "AK suited + offsuit" },
        .{ .range_str = "AK", .expected_count = 16, .description = "AK any (both s+o)" },
        .{ .range_str = "AA,AKs,AKo", .expected_count = 22, .description = "Mix: AA(6) + AKs(4) + AKo(12)" },
        .{ .range_str = "AA,AK", .expected_count = 22, .description = "Mix: AA(6) + AK(16)" },
        .{ .range_str = "KK,KQo,AJo", .expected_count = 30, .description = "KK(6) + KQo(12) + AJo(12)" },
    };

    for (test_cases) |test_case| {
        var range = parseRange(test_case.range_str, allocator) catch |err| {
            std.debug.print("FAIL: {s} - Parse error: {}\n", .{ test_case.description, err });
            try std.testing.expect(false);
            continue;
        };
        defer range.deinit();

        const actual_count = range.handCount();
        if (actual_count != test_case.expected_count) {
            std.debug.print("FAIL: {s} - Expected {} hands, got {}\n", .{ test_case.description, test_case.expected_count, actual_count });
        }
        try std.testing.expectEqual(test_case.expected_count, actual_count);
    }
}

test "parseCards delegation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test that we properly use poker.parseCards
    const single_card = try poker.parseCards("As", allocator);
    defer allocator.free(single_card);
    try std.testing.expect(single_card.len == 1);
    try std.testing.expect(single_card[0].getRank() == 14);
    try std.testing.expect(single_card[0].getSuit() == 1); // spades

    const hole_cards = try poker.parseCards("AhAs", allocator);
    defer allocator.free(hole_cards);
    try std.testing.expect(hole_cards.len == 2);
    try std.testing.expect(hole_cards[0].getRank() == 14);
    try std.testing.expect(hole_cards[1].getRank() == 14);

    const board_cards = try poker.parseCards("AdKh7s", allocator);
    defer allocator.free(board_cards);
    try std.testing.expect(board_cards.len == 3);
    try std.testing.expect(board_cards[0].getRank() == 14); // Ad
    try std.testing.expect(board_cards[1].getRank() == 13); // Kh
    try std.testing.expect(board_cards[2].getRank() == 7); // 7s
}
