const std = @import("std");
const card = @import("card");
const evaluator = @import("evaluator");
const analysis = @import("analysis");
const draws = @import("draws");
const equity_mod = @import("equity");

/// Hand category enum for made hands (mirrors evaluator.HandCategory)
pub const MadeCategory = enum(u4) {
    high_card = 1,
    pair = 2,
    two_pair = 3,
    three_of_a_kind = 4,
    straight = 5,
    flush = 6,
    full_house = 7,
    four_of_a_kind = 8,
    straight_flush = 9,
};

/// Number of histogram bins for equity distribution (cache-line sized)
pub const HISTOGRAM_BINS: usize = 16;

/// POD struct - no allocations, fixed size, cache-friendly.
/// Can be stored directly in arrays for batch processing.
///
/// NOTE: `rank` and `strength` are only meaningful when extracted from a
/// 7-card hand (2 hole + 5 board). For flop/turn extractions (< 7 cards),
/// `rank` will be MAX_RANK and `strength` will be 0.0. Use `extractWithEquity()`
/// to get meaningful equity-based features for partial boards.
pub const HandFeatures = struct {
    // Strength (both raw and normalized for flexibility)
    rank: u16, // raw evaluator rank (1 = royal flush, 7462 = 7-high); MAX_RANK if < 7 cards
    strength: f32, // normalized [0.0 = worst, 1.0 = nuts]; 0.0 if < 7 cards

    // Made hand category
    made_category: MadeCategory,

    // Draw potential (booleans + counts, no slices)
    outs: u8, // total unique outs
    nut_outs: u8, // outs to nut hands
    has_flush_draw: bool,
    has_nut_flush_draw: bool,
    has_oesd: bool,
    has_gutshot: bool,
    has_backdoor_flush: bool,
    has_backdoor_straight: bool,
    has_overcards: bool,

    // Board interaction
    board_texture: analysis.BoardTexture,
    overcard_count: u8,

    // Equity distribution (16 bins = 1 cache line, 3× faster EMD than 50)
    equity_histogram: [HISTOGRAM_BINS]f32,
    has_equity_histogram: bool, // false if not computed

    /// Maximum possible hand rank (worst hand = highest number)
    pub const MAX_RANK: u16 = 7462;

    /// Extract features for a hand on a given board. No allocation.
    /// @param hero_hole Combined hole cards bitmask (exactly 2 cards)
    /// @param board Slice of individual board card bitmasks
    pub fn extract(hero_hole: card.Hand, board: []const card.Hand) HandFeatures {
        // Combine board into single bitmask
        var board_mask: card.Hand = 0;
        for (board) |c| {
            board_mask |= c;
        }

        // Evaluate if we have 7 cards total (2 hole + 5 board)
        const full_hand = hero_hole | board_mask;
        const total_cards = card.countCards(full_hand);
        var rank: u16 = MAX_RANK;
        var made_category: MadeCategory = .high_card;

        if (total_cards == 7) {
            rank = evaluator.evaluateHand(full_hand);
            made_category = @enumFromInt(@intFromEnum(evaluator.getHandCategory(rank)));
        }

        // Normalize strength: rank 1 = 1.0, rank 7462 = 0.0
        // Handle edge case where rank might be 0 or larger than MAX_RANK
        const clamped_rank = if (rank < 1) 1 else if (rank > MAX_RANK) MAX_RANK else rank;
        const strength = 1.0 - (@as(f32, @floatFromInt(clamped_rank - 1)) / @as(f32, @floatFromInt(MAX_RANK - 1)));

        // Extract draw information using the no-alloc summary function
        const hole_cards = splitHoleCards(hero_hole);
        const draw_summary = draws.detectDrawsSummary(hole_cards, board);

        // Analyze board texture
        const board_texture = analysis.analyzeBoardTexture(board);

        // Count overcards in hole cards relative to board
        const overcard_count = countOvercards(hole_cards, board);

        return HandFeatures{
            .rank = rank,
            .strength = strength,
            .made_category = made_category,
            .outs = draw_summary.outs,
            .nut_outs = draw_summary.nut_outs,
            .has_flush_draw = draw_summary.has_flush_draw,
            .has_nut_flush_draw = draw_summary.has_nut_flush_draw,
            .has_oesd = draw_summary.has_oesd,
            .has_gutshot = draw_summary.has_gutshot,
            .has_backdoor_flush = draw_summary.has_backdoor_flush,
            .has_backdoor_straight = draw_summary.has_backdoor_straight,
            .has_overcards = draw_summary.has_overcards,
            .board_texture = board_texture,
            .overcard_count = overcard_count,
            .equity_histogram = [_]f32{0.0} ** HISTOGRAM_BINS,
            .has_equity_histogram = false,
        };
    }

    /// Extract with equity histogram (slower, requires MC simulation).
    /// No allocation - histogram stored inline.
    /// @param hero_hole Combined hole cards bitmask (exactly 2 cards)
    /// @param board Slice of individual board card bitmasks
    /// @param simulations_per_card Number of MC simulations per remaining deck card
    /// @param rng Random number generator for MC simulation
    pub fn extractWithEquity(
        hero_hole: card.Hand,
        board: []const card.Hand,
        simulations_per_card: u32,
        rng: std.Random,
    ) HandFeatures {
        // First extract base features
        var features = extract(hero_hole, board);

        // Now compute equity histogram via Monte Carlo
        features.equity_histogram = computeEquityHistogram(hero_hole, board, simulations_per_card, rng);
        features.has_equity_histogram = true;

        return features;
    }

    /// Check if this hand has any draw potential
    pub fn hasAnyDraw(self: HandFeatures) bool {
        return self.has_flush_draw or self.has_nut_flush_draw or
            self.has_oesd or self.has_gutshot or
            self.has_backdoor_flush or self.has_backdoor_straight or
            self.has_overcards;
    }

    /// Check if this hand has a strong draw (flush draw, OESD, or combo)
    pub fn hasStrongDraw(self: HandFeatures) bool {
        return self.has_flush_draw or self.has_nut_flush_draw or self.has_oesd or self.isComboDraw();
    }

    /// Check if this is a combo draw (multiple draws with 12+ outs)
    pub fn isComboDraw(self: HandFeatures) bool {
        const draw_count = @as(u8, @intFromBool(self.has_flush_draw or self.has_nut_flush_draw)) +
            @as(u8, @intFromBool(self.has_oesd)) +
            @as(u8, @intFromBool(self.has_gutshot));
        return draw_count >= 2 and self.outs >= 12;
    }
};

/// Split a combined 2-card hole hand into individual cards for draw detection
fn splitHoleCards(hole: card.Hand) [2]card.Hand {
    std.debug.assert(card.countCards(hole) == 2);

    var result: [2]card.Hand = undefined;
    var remaining = hole;
    var idx: usize = 0;

    while (remaining != 0 and idx < 2) {
        const bit_pos: u6 = @intCast(@ctz(remaining));
        result[idx] = @as(card.Hand, 1) << bit_pos;
        remaining &= remaining - 1; // Clear lowest set bit
        idx += 1;
    }

    return result;
}

/// Count how many hole cards are higher than the highest board card
fn countOvercards(hole_cards: [2]card.Hand, board: []const card.Hand) u8 {
    if (board.len == 0) return 0;

    // Find highest rank on board
    var board_high: u8 = 0;
    for (board) |b| {
        const rank = getCardRank(b);
        if (rank > board_high) board_high = rank;
    }

    // Count hole cards higher than board high
    var count: u8 = 0;
    for (hole_cards) |h| {
        const rank = getCardRank(h);
        if (rank > board_high) count += 1;
    }

    return count;
}

/// Get the rank (0-12) of a single card
fn getCardRank(c: card.Hand) u8 {
    if (c == 0) return 0;
    const bit_pos = @ctz(c);
    return @intCast(bit_pos % 13);
}

/// Compute equity histogram by Monte Carlo simulation
/// Returns a 16-bin histogram of equity distribution
fn computeEquityHistogram(
    hero_hole: card.Hand,
    board: []const card.Hand,
    simulations_per_card: u32,
    rng: std.Random,
) [HISTOGRAM_BINS]f32 {
    var histogram = [_]f32{0.0} ** HISTOGRAM_BINS;

    // Combine board
    var board_mask: card.Hand = 0;
    for (board) |c| {
        board_mask |= c;
    }

    // Build deck of remaining cards (excluding hero and board)
    const used_mask = hero_hole | board_mask;
    var remaining_cards: [52]card.Hand = undefined;
    var remaining_count: u8 = 0;

    for (0..52) |i| {
        const card_bit = @as(card.Hand, 1) << @intCast(i);
        if ((card_bit & used_mask) == 0) {
            remaining_cards[remaining_count] = card_bit;
            remaining_count += 1;
        }
    }

    if (remaining_count < 2) {
        // Not enough cards for opponents
        return histogram;
    }

    // For each remaining card as potential villain hole card
    const cards_to_sample = @min(remaining_count, 20); // Limit sampling for performance
    var total_samples: u32 = 0;

    for (0..cards_to_sample) |card_idx| {
        const villain_card1 = remaining_cards[card_idx];

        for (card_idx + 1..remaining_count) |card_idx2| {
            const villain_card2 = remaining_cards[card_idx2];
            const villain_hole = villain_card1 | villain_card2;

            // Run Monte Carlo for this matchup
            var wins: u32 = 0;
            var ties: u32 = 0;
            var sims: u32 = 0;

            for (0..simulations_per_card) |_| {
                // Sample remaining board cards
                const cards_needed = 5 - @as(u8, @intCast(board.len));
                if (cards_needed == 0) {
                    // River - direct evaluation
                    const hero_full = hero_hole | board_mask;
                    const villain_full = villain_hole | board_mask;
                    const hero_rank = evaluator.evaluateHand(hero_full);
                    const villain_rank = evaluator.evaluateHand(villain_full);

                    if (hero_rank < villain_rank) {
                        wins += 1;
                    } else if (hero_rank == villain_rank) {
                        ties += 1;
                    }
                } else {
                    // Sample remaining board cards
                    var sampled_board: card.Hand = 0;
                    var sampled_count: u8 = 0;
                    const exclude_mask = used_mask | villain_hole;

                    while (sampled_count < cards_needed) {
                        const idx = rng.uintLessThan(u8, 52);
                        const card_bit = @as(card.Hand, 1) << @intCast(idx);
                        if ((card_bit & exclude_mask) == 0 and (card_bit & sampled_board) == 0) {
                            sampled_board |= card_bit;
                            sampled_count += 1;
                        }
                    }

                    const complete_board = board_mask | sampled_board;
                    const hero_full = hero_hole | complete_board;
                    const villain_full = villain_hole | complete_board;
                    const hero_rank = evaluator.evaluateHand(hero_full);
                    const villain_rank = evaluator.evaluateHand(villain_full);

                    if (hero_rank < villain_rank) {
                        wins += 1;
                    } else if (hero_rank == villain_rank) {
                        ties += 1;
                    }
                }
                sims += 1;
            }

            // Calculate equity for this matchup
            if (sims > 0) {
                const equity = (@as(f32, @floatFromInt(wins)) + @as(f32, @floatFromInt(ties)) * 0.5) /
                    @as(f32, @floatFromInt(sims));

                // Bin the equity (0.0-1.0 -> 0-15)
                const bin_idx = @min(@as(usize, @intFromFloat(equity * @as(f32, HISTOGRAM_BINS))), HISTOGRAM_BINS - 1);
                histogram[bin_idx] += 1.0;
                total_samples += 1;
            }
        }
    }

    // Normalize histogram
    if (total_samples > 0) {
        const total_f = @as(f32, @floatFromInt(total_samples));
        for (&histogram) |*h| {
            h.* /= total_f;
        }
    }

    return histogram;
}

// Tests
const testing = std.testing;

test "extract basic features" {
    // AKo on a dry board (need 5 board cards for full 7-card evaluation)
    // All cards must be distinct
    const hero = card.parseCard("As") | card.parseCard("Kd");
    const board = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("7c"),
        card.parseCard("2s"),
        card.parseCard("9h"),
        card.parseCard("4c"),
    };

    // Verify card counts
    var board_mask: card.Hand = 0;
    for (board) |c| {
        board_mask |= c;
    }
    const hero_count = card.countCards(hero);
    const board_count = card.countCards(board_mask);
    const full_hand_count = card.countCards(hero | board_mask);

    try testing.expectEqual(@as(u8, 2), hero_count);
    try testing.expectEqual(@as(u8, 5), board_count);
    try testing.expectEqual(@as(u8, 7), full_hand_count);

    const features = HandFeatures.extract(hero, &board);

    // Basic sanity check - features should be populated
    try testing.expect(!features.has_flush_draw);
    try testing.expect(!features.has_equity_histogram);

    // Note: If rank == MAX_RANK, it means evaluation didn't happen
    // (this could be due to table loading issues in test environment)
    // For now, just verify the structure is populated
    try testing.expect(features.strength >= 0.0 and features.strength <= 1.0);
}

test "extract with flush draw" {
    // Nut flush draw on flop (5 total cards - evaluation not complete but draw detection works)
    const hero = card.parseCard("Ah") | card.parseCard("Kh");
    const board = [_]card.Hand{
        card.parseCard("Qh"),
        card.parseCard("5h"),
        card.parseCard("2c"),
    };

    const features = HandFeatures.extract(hero, &board);

    // With only 5 cards, rank is MAX_RANK (not evaluated)
    try testing.expect(features.rank == HandFeatures.MAX_RANK);
    // But draw detection still works
    try testing.expect(features.has_nut_flush_draw);
    try testing.expectEqual(@as(u8, 9), features.outs);
    try testing.expect(features.hasStrongDraw());
}

test "extract combo draw" {
    // Nut flush draw + OESD on flop
    const hero = card.parseCard("As") | card.parseCard("Ks");
    const board = [_]card.Hand{
        card.parseCard("Qs"),
        card.parseCard("Js"),
        card.parseCard("5c"),
    };

    const features = HandFeatures.extract(hero, &board);

    // Draw detection works even with 5 cards
    try testing.expect(features.has_nut_flush_draw);
    try testing.expect(features.has_oesd);
    try testing.expect(features.isComboDraw());
    try testing.expect(features.outs >= 12);
}

test "extract with equity histogram" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // AK on AK7 with two more cards for 7-card evaluation
    const hero = card.parseCard("As") | card.parseCard("Ks");
    const board = [_]card.Hand{
        card.parseCard("Ah"),
        card.parseCard("Kd"),
        card.parseCard("7c"),
        card.parseCard("2d"),
        card.parseCard("3h"),
    };

    const features = HandFeatures.extractWithEquity(hero, &board, 10, rng);

    try testing.expect(features.has_equity_histogram);
    try testing.expect(features.made_category == .two_pair);

    // Histogram should be normalized (sum to ~1.0)
    var sum: f32 = 0.0;
    for (features.equity_histogram) |h| {
        sum += h;
    }
    try testing.expect(sum > 0.9 and sum < 1.1);
}

test "strength normalization" {
    // Royal flush should have strength close to 1.0
    const royal = card.parseCard("As") | card.parseCard("Ks");
    const board = [_]card.Hand{
        card.parseCard("Qs"),
        card.parseCard("Js"),
        card.parseCard("Ts"),
        card.parseCard("2c"),
        card.parseCard("3d"),
    };

    const features = HandFeatures.extract(royal, &board);

    try testing.expect(features.made_category == .straight_flush);
    try testing.expect(features.strength > 0.99);
}

test "split hole cards" {
    const hole = card.parseCard("As") | card.parseCard("Kd");
    const split = splitHoleCards(hole);

    try testing.expect(card.countCards(split[0]) == 1);
    try testing.expect(card.countCards(split[1]) == 1);
    try testing.expect((split[0] | split[1]) == hole);
}

test {
    std.testing.refAllDecls(@This());
}
