const std = @import("std");
const card = @import("card");
const hand = @import("hand");
const equity = @import("equity");

/// Range representation and poker notation parsing
/// Handles range strings like "AA,KK,AKs,AKo" and converts them to hand combinations
/// Notation parsing is implemented inline for efficient direct insertion into range HashMap
///
/// ## Supported Range Notation
///
/// ### 1. Pocket Pairs (6 combinations each)
/// - "AA" - All pocket aces: AcAd, AcAh, AcAs, AdAh, AdAs, AhAs
/// - "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55", "44", "33", "22"
///
/// ### 2. Suited Hands (4 combinations each)
/// - "AKs" - All suited AK: AcKc, AdKd, AhKh, AsKs
/// - Works with any two different ranks: "KQs", "JTs", "98s", "72s", etc.
///
/// ### 3. Offsuit Hands (12 combinations each)
/// - "AKo" - All offsuit AK: AcKd, AcKh, AcKs, AdKc, AdKh, AdKs, AhKc, AhKd, AhKs, AsKc, AsKd, AsKh
/// - Works with any two different ranks: "KQo", "JTo", "98o", "72o", etc.
///
/// ### 4. Unpaired Without Modifier (16 combinations = suited + offsuit)
/// - "AK" - All AK hands (both suited and offsuit)
/// - Expands to all 16 possible combinations of two different ranks
///
/// ### 5. Specific Card Combinations (1 combination each)
/// - "AhAs" - Exact hand: Ace of hearts and Ace of spades
/// - "KdKc" - Exact hand: King of diamonds and King of clubs
/// - Format: Two cards with explicit suits (rank+suit for each card)
/// - Suits: h=hearts, d=diamonds, c=clubs, s=spades
///
/// ### 6. Comma-Separated Ranges (mix any of the above)
/// - "AA,KK,AKs" - All AA, all KK, and all suited AK (6+6+4 = 16 combos)
/// - "AhAs,KdKc,QQ" - Two specific hands plus all QQ (1+1+6 = 8 combos)
/// - "AA,AKs,AKo,AhAs" - Mix range notation with specific hands
/// - Spaces are trimmed, so "AA, KK, QQ" works fine
///
/// ### Notes
/// - Notation is case-insensitive: "aa", "AA", "Aa" all work
/// - Ranks: A, K, Q, J, T (ten), 9, 8, 7, 6, 5, 4, 3, 2
/// - Pocket pairs cannot have suit modifiers ("AAs" is invalid)

// Re-export core types for convenience
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;

/// Efficient representation of a poker hand range
/// Uses HashMap with combined Hand (u64) as key for performance
pub const Range = struct {
    /// Map from combined hand to probability
    hands: std.AutoHashMap(Hand, f32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Range {
        return Range{
            .hands = std.AutoHashMap(Hand, f32).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Range) void {
        self.hands.deinit();
    }

    /// Add a hand to the range with given probability
    pub fn addHand(self: *Range, hole_hand: [2]Hand, probability: f32) !void {
        const combined = hand.combine(hole_hand);
        try self.hands.put(combined, probability);
    }

    /// Add a hand using poker notation (e.g., "AA", "AKs", "AKo", "AhAs")
    /// For pocket pairs like "AA", this adds ALL possible combinations (6 for AA)
    /// For suited/offsuit like "AKs"/"AKo", this adds all combinations
    /// For unpaired notation like "AK", adds all combinations (16 total)
    /// For specific cards like "AhAs", adds just that single combination
    pub fn addHandNotation(self: *Range, notation: []const u8, probability: f32) !void {
        // Handle specific card notation (4 chars like "AhAs")
        if (notation.len == 4) {
            const parsed_hand = hand.maybeParseHand(notation) catch return error.InvalidSpecificHand;
            if (card.countCards(parsed_hand) != 2) return error.InvalidSpecificHand;
            const hole_hand = hand.split(parsed_hand);
            try self.addHand(hole_hand, probability);
            return;
        }

        if (notation.len < 2 or notation.len > 3) return error.InvalidNotation;

        const rank1 = hand.parseRank(notation[0]) orelse return error.InvalidRank;
        const rank2 = hand.parseRank(notation[1]) orelse return error.InvalidRank;

        if (notation.len == 2) {
            if (rank1 == rank2) {
                // Pocket pair - add all 6 combinations
                const combinations = try hand.generatePocketPair(rank1, self.allocator);
                defer self.allocator.free(combinations);

                for (combinations) |combo| {
                    try self.addHand(combo, probability);
                }
            } else {
                // Unpaired hand without modifier - add all 16 combinations (suited + offsuit)
                const combinations = try hand.generateAllCombinations(rank1, rank2, self.allocator);
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
                    const combinations = try hand.generateSuitedCombinations(rank1, rank2, self.allocator);
                    defer self.allocator.free(combinations);

                    for (combinations) |combo| {
                        try self.addHand(combo, probability);
                    }
                },
                'o', 'O' => {
                    // Offsuit - add 12 combinations
                    const combinations = try hand.generateOffsuitCombinations(rank1, rank2, self.allocator);
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
        const combined = hand.combine(hole_hand);
        return self.hands.get(combined) orelse 0.0;
    }

    /// Get total number of hand combinations in range
    pub fn handCount(self: *const Range) u32 {
        return @intCast(self.hands.count());
    }

    /// Iterator for efficient range traversal (returns split hands for API compatibility)
    pub const Iterator = struct {
        inner: std.AutoHashMap(Hand, f32).Iterator,

        pub fn next(self: *Iterator) ?struct { hand: [2]Hand, probability: f32 } {
            if (self.inner.next()) |entry| {
                const combined = entry.key_ptr.*;
                const probability = entry.value_ptr.*;

                return .{ .hand = hand.split(combined), .probability = probability };
            }
            return null;
        }
    };

    pub fn iterator(self: *const Range) Iterator {
        return Iterator{ .inner = self.hands.iterator() };
    }

    /// Internal iterator that returns combined hands for performance
    const CombinedIterator = struct {
        inner: std.AutoHashMap(Hand, f32).Iterator,

        pub fn next(self: *CombinedIterator) ?struct { hand: Hand, probability: f32 } {
            if (self.inner.next()) |entry| {
                return .{ .hand = entry.key_ptr.*, .probability = entry.value_ptr.* };
            }
            return null;
        }
    };

    fn combinedIterator(self: *const Range) CombinedIterator {
        return CombinedIterator{ .inner = self.hands.iterator() };
    }

    /// Sample a random hand from this range using weighted probability
    pub fn sample(self: *const Range, rng: std.Random) [2]Hand {
        return hand.split(self.sampleCombined(rng));
    }

    /// Sample a random combined hand from this range (optimized for hot paths)
    fn sampleCombined(self: *const Range, rng: std.Random) Hand {
        // Calculate total weight
        var total_weight: f64 = 0;
        var iter = self.combinedIterator();
        while (iter.next()) |entry| {
            total_weight += entry.probability;
        }

        // Sample based on weight
        const target = rng.float(f64) * total_weight;
        var cumulative: f64 = 0;

        var sample_iter = self.combinedIterator();
        while (sample_iter.next()) |entry| {
            cumulative += entry.probability;
            if (cumulative >= target) {
                return entry.hand;
            }
        }

        // Fallback - return first hand (should not happen with proper weights)
        var fallback_iter = self.combinedIterator();
        if (fallback_iter.next()) |entry| {
            return entry.hand;
        }

        // This should never happen with non-empty range
        unreachable;
    }

    /// Calculate exact range vs range equity
    pub fn equityExact(self: *const Range, opponent: *const Range, board: []const Hand, allocator: std.mem.Allocator) !RangeEquityResult {
        // Handle empty ranges
        if (self.handCount() == 0 or opponent.handCount() == 0) {
            return RangeEquityResult{
                .hero_equity = 0.0,
                .villain_equity = 0.0,
                .total_simulations = 0,
                .hero_wins = 0,
                .ties = 0,
                .hero_losses = 0,
            };
        }

        var total_hero_equity: f64 = 0.0;
        var total_weight: f64 = 0.0;
        var total_combinations: u32 = 0;

        // Accumulators for win/tie/loss
        var total_hero_wins: u32 = 0;
        var total_ties: u32 = 0;
        var total_hero_losses: u32 = 0;

        // Combine board cards once
        var board_hand: Hand = 0;
        for (board) |board_card| {
            board_hand |= board_card;
        }

        // Use combined iterator for better performance
        var hero_iter = self.combinedIterator();
        while (hero_iter.next()) |hero_entry| {
            var villain_iter = opponent.combinedIterator();
            while (villain_iter.next()) |villain_entry| {
                // Check for card conflicts using combined hands
                if ((hero_entry.hand & villain_entry.hand) != 0 or
                    (hero_entry.hand & board_hand) != 0 or
                    (villain_entry.hand & board_hand) != 0)
                {
                    continue;
                }

                // Calculate weight for this combination
                const weight = hero_entry.probability * villain_entry.probability;
                total_weight += weight;

                // Calculate equity for this matchup (already combined)
                const result = try equity.exact(hero_entry.hand, villain_entry.hand, board, allocator);

                // Accumulate weighted equity
                total_hero_equity += result.equity() * weight;
                total_combinations += 1;

                // Accumulate win/tie/loss counts
                total_hero_wins += result.wins;
                total_ties += result.ties;
                total_hero_losses += (result.total_simulations - result.wins - result.ties);
            }
        }

        const hero_equity = if (total_weight > 0) total_hero_equity / total_weight else 0.0;

        return RangeEquityResult{
            .hero_equity = hero_equity,
            .villain_equity = 1.0 - hero_equity,
            .total_simulations = total_hero_wins + total_ties + total_hero_losses,
            .hero_wins = total_hero_wins,
            .ties = total_ties,
            .hero_losses = total_hero_losses,
        };
    }

    /// Calculate detailed exact range vs range equity with hand category tracking
    pub fn equityExactWithCategories(self: *const Range, opponent: *const Range, board: []const Hand, allocator: std.mem.Allocator) !DetailedRangeEquityResult {
        // Handle empty ranges
        if (self.handCount() == 0 or opponent.handCount() == 0) {
            return DetailedRangeEquityResult{
                .hero_equity = 0.0,
                .villain_equity = 0.0,
                .total_simulations = 0,
                .hero_wins = 0,
                .ties = 0,
                .hero_losses = 0,
            };
        }

        var total_hero_equity: f64 = 0.0;
        var total_weight: f64 = 0.0;
        var total_combinations: u32 = 0;

        // Accumulators for win/tie/loss (u64 to prevent overflow with large ranges)
        var total_hero_wins: u64 = 0;
        var total_ties: u64 = 0;
        var total_hero_losses: u64 = 0;

        // Hand category accumulators
        var hero_categories = equity.HandCategories{};
        var villain_categories = equity.HandCategories{};

        // Combine board cards once
        var board_hand: Hand = 0;
        for (board) |board_card| {
            board_hand |= board_card;
        }

        // Use combined iterator for better performance
        var hero_iter = self.combinedIterator();
        while (hero_iter.next()) |hero_entry| {
            var villain_iter = opponent.combinedIterator();
            while (villain_iter.next()) |villain_entry| {
                // Check for card conflicts using combined hands
                if ((hero_entry.hand & villain_entry.hand) != 0 or
                    (hero_entry.hand & board_hand) != 0 or
                    (villain_entry.hand & board_hand) != 0)
                {
                    continue;
                }

                // Calculate weight for this combination
                const weight = hero_entry.probability * villain_entry.probability;
                total_weight += weight;

                // Calculate detailed equity for this matchup (already combined)
                const result = try equity.exactWithCategories(hero_entry.hand, villain_entry.hand, board, allocator);

                // Accumulate weighted equity
                total_hero_equity += result.equity() * weight;
                total_combinations += 1;

                // Accumulate win/tie/loss counts
                total_hero_wins += result.wins;
                total_ties += result.ties;
                total_hero_losses += (result.total_simulations - result.wins - result.ties);

                // Accumulate hand categories (unwrap optional since exactWithCategories always populates them)
                const h1_cats = result.hand1_categories.?;
                const h2_cats = result.hand2_categories.?;

                hero_categories.high_card += h1_cats.high_card;
                hero_categories.pair += h1_cats.pair;
                hero_categories.two_pair += h1_cats.two_pair;
                hero_categories.three_of_a_kind += h1_cats.three_of_a_kind;
                hero_categories.straight += h1_cats.straight;
                hero_categories.flush += h1_cats.flush;
                hero_categories.full_house += h1_cats.full_house;
                hero_categories.four_of_a_kind += h1_cats.four_of_a_kind;
                hero_categories.straight_flush += h1_cats.straight_flush;
                hero_categories.total += h1_cats.total;

                villain_categories.high_card += h2_cats.high_card;
                villain_categories.pair += h2_cats.pair;
                villain_categories.two_pair += h2_cats.two_pair;
                villain_categories.three_of_a_kind += h2_cats.three_of_a_kind;
                villain_categories.straight += h2_cats.straight;
                villain_categories.flush += h2_cats.flush;
                villain_categories.full_house += h2_cats.full_house;
                villain_categories.four_of_a_kind += h2_cats.four_of_a_kind;
                villain_categories.straight_flush += h2_cats.straight_flush;
                villain_categories.total += h2_cats.total;
            }
        }

        const hero_equity = if (total_weight > 0) total_hero_equity / total_weight else 0.0;

        return DetailedRangeEquityResult{
            .hero_equity = hero_equity,
            .villain_equity = 1.0 - hero_equity,
            .total_simulations = total_hero_wins + total_ties + total_hero_losses,
            .hero_wins = total_hero_wins,
            .ties = total_ties,
            .hero_losses = total_hero_losses,
            .hero_categories = hero_categories,
            .villain_categories = villain_categories,
        };
    }

    /// Calculate range vs range equity using Monte Carlo simulation
    pub fn equityMonteCarlo(self: *const Range, opponent: *const Range, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !RangeEquityResult {
        // Handle empty ranges
        if (self.handCount() == 0 or opponent.handCount() == 0) {
            return RangeEquityResult{
                .hero_equity = 0.0,
                .villain_equity = 0.0,
                .total_simulations = 0,
                .hero_wins = 0,
                .ties = 0,
                .hero_losses = 0,
            };
        }

        var hero_wins: u32 = 0;
        var ties: u32 = 0;
        var hero_losses: u32 = 0;
        var valid_simulations: u32 = 0;

        // Combine board cards once
        var board_hand: Hand = 0;
        for (board) |board_card| {
            board_hand |= board_card;
        }

        var i: u32 = 0;
        while (i < simulations) : (i += 1) {
            // Sample combined hands directly (no split needed)
            const hero_combined = self.sampleCombined(rng);
            const villain_combined = opponent.sampleCombined(rng);

            // Check for card conflicts using combined hands
            if ((hero_combined & villain_combined) != 0 or
                (hero_combined & board_hand) != 0 or
                (villain_combined & board_hand) != 0)
            {
                continue;
            }

            // Run Monte Carlo simulation for this matchup
            const result = try equity.monteCarlo(
                hero_combined,
                villain_combined,
                board,
                1, // Single simulation since we're already sampling
                rng,
                allocator,
            );

            valid_simulations += 1;

            if (result.wins > 0) {
                hero_wins += 1;
            } else if (result.ties > 0) {
                ties += 1;
            } else {
                hero_losses += 1;
            }
        }

        // Handle case where all simulations were skipped due to conflicts
        if (valid_simulations == 0) {
            return RangeEquityResult{
                .hero_equity = 0.0,
                .villain_equity = 0.0,
                .total_simulations = 0,
                .hero_wins = 0,
                .ties = 0,
                .hero_losses = 0,
            };
        }

        const hero_equity = (@as(f64, @floatFromInt(hero_wins)) + @as(f64, @floatFromInt(ties)) * 0.5) / @as(f64, @floatFromInt(valid_simulations));

        return RangeEquityResult{
            .hero_equity = hero_equity,
            .villain_equity = 1.0 - hero_equity,
            .total_simulations = valid_simulations,
            .hero_wins = hero_wins,
            .ties = ties,
            .hero_losses = hero_losses,
        };
    }
};

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

    // Win/tie/loss breakdown
    hero_wins: u32,
    ties: u32,
    hero_losses: u32,

    pub fn sum(self: RangeEquityResult) f64 {
        return self.hero_equity + self.villain_equity;
    }

    pub fn winRate(self: RangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hero_wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn tieRate(self: RangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ties)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn lossRate(self: RangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hero_losses)) / @as(f64, @floatFromInt(self.total_simulations));
    }
};

/// Detailed range equity result with hand category tracking
pub const DetailedRangeEquityResult = struct {
    hero_equity: f64,
    villain_equity: f64,
    total_simulations: u64,

    // Win/tie/loss breakdown (u64 to handle large range enumerations)
    hero_wins: u64,
    ties: u64,
    hero_losses: u64,

    // Hand category tracking
    hero_categories: equity.HandCategories = equity.HandCategories{},
    villain_categories: equity.HandCategories = equity.HandCategories{},

    pub fn sum(self: DetailedRangeEquityResult) f64 {
        return self.hero_equity + self.villain_equity;
    }

    pub fn winRate(self: DetailedRangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hero_wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn tieRate(self: DetailedRangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.ties)) / @as(f64, @floatFromInt(self.total_simulations));
    }

    pub fn lossRate(self: DetailedRangeEquityResult) f64 {
        if (self.total_simulations == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hero_losses)) / @as(f64, @floatFromInt(self.total_simulations));
    }
};

/// Calculate exact range vs range equity
/// Deprecated: Use Range.equityExact() instead
pub fn calculateRangeEquityExact(hero_range: *const Range, villain_range: *const Range, board: []const Hand, allocator: std.mem.Allocator) !RangeEquityResult {
    return hero_range.equityExact(villain_range, board, allocator);
}

/// Calculate range vs range equity using Monte Carlo simulation
/// Deprecated: Use Range.equityMonteCarlo() instead
pub fn calculateRangeEquityMonteCarlo(hero_range: *const Range, villain_range: *const Range, board: []const Hand, simulations: u32, rng: std.Random, allocator: std.mem.Allocator) !RangeEquityResult {
    return hero_range.equityMonteCarlo(villain_range, board, simulations, rng, allocator);
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

test "combine and split round-trip" {
    const hand1 = [2]Hand{
        card.makeCard(.hearts, .ace), // Ace of hearts
        card.makeCard(.spades, .ace), // Ace of spades
    };

    const combined = hand.combine(hand1);
    const reconstructed = hand.split(combined);

    // Check that the reconstructed bits match the original bits
    const original_bits = hand.combine(hand1);
    const reconstructed_bits = hand.combine(reconstructed);
    try testing.expect(original_bits == reconstructed_bits);
}

test "range notation parsing" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test parseRank function (now in hand.zig)
    try testing.expectEqual(@as(?Rank, .two), hand.parseRank('2'));
    try testing.expectEqual(@as(?Rank, .ten), hand.parseRank('T'));
    try testing.expectEqual(@as(?Rank, .ace), hand.parseRank('A'));
    try testing.expectEqual(@as(?Rank, null), hand.parseRank('X'));

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

test "Range.equityExact AA vs KK on turn" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    // Turn board - only 44 rivers per matchup = 1,584 total evaluations
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
        card.makeCard(.clubs, .two),
    };

    const result = try hero_range.equityExact(&villain_range, &board, allocator);

    // AA should beat KK with this board
    // 6 AA combos * 6 KK combos = 36 matchups (some may conflict with board)
    try testing.expect(result.hero_equity > 0.90);
}

test "Range.equityExact rate methods return valid values" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    // Turn board for fast test
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
        card.makeCard(.clubs, .two),
    };

    const result = try hero_range.equityExact(&villain_range, &board, allocator);

    // Verify rate methods return values in [0, 1] range (not > 1)
    try testing.expect(result.winRate() >= 0.0 and result.winRate() <= 1.0);
    try testing.expect(result.tieRate() >= 0.0 and result.tieRate() <= 1.0);
    try testing.expect(result.lossRate() >= 0.0 and result.lossRate() <= 1.0);

    // Verify rates sum to 1.0 (within floating point tolerance)
    const rate_sum = result.winRate() + result.tieRate() + result.lossRate();
    try testing.expect(rate_sum >= 0.99 and rate_sum <= 1.01);

    // Verify total_simulations matches sum of win/tie/loss
    try testing.expect(result.total_simulations == result.hero_wins + result.ties + result.hero_losses);
}

test "Range.equityMonteCarlo AA vs 22" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("22", 1.0);

    const result = try hero_range.equityMonteCarlo(&villain_range, &.{}, 10000, rng, allocator);

    // AA should dominate 22 (~85% equity)
    try testing.expect(result.hero_equity > 0.80);
    try testing.expect(result.hero_equity < 0.90);
}

test "Range.equityMonteCarlo with board" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    // Empty board - AA should dominate KK
    const result = try hero_range.equityMonteCarlo(&villain_range, &.{}, 10000, rng, allocator);

    // AA should beat KK ~82% of the time
    try testing.expect(result.hero_equity > 0.75);
    try testing.expect(result.hero_equity < 0.90);
}

test "backward compatibility calculateRangeEquityExact" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    // Turn board for fast test
    const board = [_]Hand{
        card.makeCard(.spades, .seven),
        card.makeCard(.hearts, .eight),
        card.makeCard(.diamonds, .nine),
        card.makeCard(.clubs, .two),
    };

    const result = try calculateRangeEquityExact(&hero_range, &villain_range, &board, allocator);

    // Should produce same results as Range.equityExact
    try testing.expect(result.hero_equity > 0.90);
}

test "backward compatibility calculateRangeEquityMonteCarlo" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    const result = try calculateRangeEquityMonteCarlo(&hero_range, &villain_range, &.{}, 5000, rng, allocator);

    // Should produce same results as Range.equityMonteCarlo
    try testing.expect(result.hero_equity > 0.75);
    try testing.expect(result.hero_equity < 0.90);
}

test "empty range should return zero equity" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("AA", 1.0);

    // Empty hero range should handle gracefully
    const result = try hero_range.equityMonteCarlo(&villain_range, &.{}, 1000, rng, allocator);
    try testing.expect(result.hero_equity == 0.0);
    try testing.expect(result.villain_equity == 0.0);
}

test "card conflict in range should be handled" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var hero_range = Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("AA", 1.0);

    // Same range should detect conflicts and skip those combos
    const result = try hero_range.equityMonteCarlo(&villain_range, &.{}, 1000, rng, allocator);

    // Should be approximately 0.5 equity since they tie when not conflicting
    // Total simulations should be less than 1000 due to skipped conflicts
    try testing.expect(result.hero_equity >= 0.0);
    try testing.expect(result.hero_equity <= 1.0);
    try testing.expect(result.total_simulations < 1000);
    try testing.expect(result.total_simulations > 0);
}

test "range with weighted probabilities" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add hands with different probabilities
    try range.addHandNotation("AA", 1.0);
    try range.addHandNotation("KK", 0.5);

    // Should have 6 + 6 = 12 total hands
    try testing.expect(range.handCount() == 12);
}

test "Range.sample returns valid hand from range" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var range = Range.init(allocator);
    defer range.deinit();
    try range.addHandNotation("AA", 1.0);

    // Sample should return one of the AA combinations
    const sampled = range.sample(rng);
    const combined = sampled[0] | sampled[1];

    // Should be exactly 2 cards
    try testing.expect(card.countCards(combined) == 2);

    // Both should be aces
    try testing.expect(card.hasCard(combined, .clubs, .ace) or
        card.hasCard(combined, .diamonds, .ace) or
        card.hasCard(combined, .hearts, .ace) or
        card.hasCard(combined, .spades, .ace));
}

test "hasCardConflict detects overlaps" {
    const hero = [2]Hand{
        card.makeCard(.clubs, .ace),
        card.makeCard(.diamonds, .ace),
    };
    const villain = [2]Hand{
        card.makeCard(.clubs, .ace), // Conflicts with hero
        card.makeCard(.hearts, .king),
    };

    try testing.expect(hand.hasCardConflict(hero, villain, &.{}));
}

test "hasCardConflict with board" {
    const hero = [2]Hand{
        card.makeCard(.clubs, .ace),
        card.makeCard(.diamonds, .ace),
    };
    const villain = [2]Hand{
        card.makeCard(.hearts, .king),
        card.makeCard(.spades, .king),
    };
    const board = [_]Hand{
        card.makeCard(.clubs, .ace), // Conflicts with hero
        card.makeCard(.hearts, .seven),
        card.makeCard(.spades, .two),
    };

    try testing.expect(hand.hasCardConflict(hero, villain, &board));
}

test "RangeEquityResult methods" {
    const result = RangeEquityResult{
        .hero_equity = 0.6,
        .villain_equity = 0.4,
        .total_simulations = 1000,
        .hero_wins = 600,
        .ties = 0,
        .hero_losses = 400,
    };

    try testing.expect(result.sum() == 1.0);
    try testing.expect(result.winRate() == 0.6);
    try testing.expect(result.lossRate() == 0.4);
}

test "invalid range notation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Test invalid notations
    try testing.expectError(error.InvalidRank, range.addHandNotation("XX", 1.0));
    try testing.expectError(error.InvalidNotation, range.addHandNotation("A", 1.0));
    try testing.expectError(error.InvalidModifier, range.addHandNotation("AKx", 1.0));
}

test "pocket pair cannot be suited" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    try testing.expectError(error.CannotBeSuited, range.addHandNotation("AAs", 1.0));
    try testing.expectError(error.CannotBeSuited, range.addHandNotation("KKo", 1.0));
}

test "specific card notation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var range = Range.init(allocator);
    defer range.deinit();

    // Add specific hole cards
    try range.addHandNotation("AhAs", 1.0);
    try testing.expect(range.handCount() == 1); // Only 1 specific combination

    // Add another specific hand
    try range.addHandNotation("KdKc", 1.0);
    try testing.expect(range.handCount() == 2);

    // Mix specific and range notation
    try range.addHandNotation("QQ", 1.0);
    try testing.expect(range.handCount() == 8); // 2 specific + 6 QQ combos
}

test "parseRange with specific cards" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test mixed notation
    var range = try parseRange("AA,KdKc,QhQs", allocator);
    defer range.deinit();

    // AA = 6 combos, KdKc = 1 combo, QhQs = 1 combo (total 8)
    try testing.expect(range.handCount() == 8);
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
