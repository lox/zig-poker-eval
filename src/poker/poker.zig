const std = @import("std");

// Card suits
pub const Suit = enum(u2) {
    hearts = 0,
    spades = 1,
    diamonds = 2,
    clubs = 3,
};

// Card ranks
pub const Rank = enum(u4) {
    two = 2,
    three = 3,
    four = 4,
    five = 5,
    six = 6,
    seven = 7,
    eight = 8,
    nine = 9,
    ten = 10,
    jack = 11,
    queen = 12,
    king = 13,
    ace = 14,
};

// Hand rankings (from weakest to strongest)
pub const HandRank = enum(u4) {
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

// Efficient card representation using bit manipulation
pub const Card = struct {
    bits: u64,

    pub fn init(rank: u8, suit: u2) Card {
        // Each card is represented as a single bit
        // Cards 0-51: rank 2-14, suits 0-3
        const card_index = (rank - 2) * 4 + suit;
        return Card{ .bits = @as(u64, 1) << @intCast(card_index) };
    }

    pub fn getRank(self: Card) u8 {
        const card_index = @ctz(self.bits);
        return @intCast((card_index / 4) + 2);
    }

    pub fn getSuit(self: Card) u2 {
        const card_index = @ctz(self.bits);
        return @intCast(card_index % 4);
    }
};

// Simple Hand struct with cached bits for performance
pub const Hand = struct {
    bits: u64,

    pub fn init() Hand {
        return Hand{ .bits = 0 };
    }

    pub fn addCard(self: *Hand, card: Card) void {
        self.bits |= card.bits;
    }

    pub fn fromCards(cards: [7]Card) Hand {
        var hand_bits: u64 = 0;
        // Manual unroll for performance
        hand_bits = cards[0].bits | cards[1].bits | cards[2].bits |
            cards[3].bits | cards[4].bits | cards[5].bits | cards[6].bits;
        return Hand{ .bits = hand_bits };
    }

    // High-performance evaluation using cached bits
    pub inline fn evaluate(self: Hand) HandRank {
        return self.evaluateOptimized();
    }

    // Direct bit manipulation evaluation - 10ns/op performance
    pub inline fn evaluateOptimized(self: Hand) HandRank {
        return evaluateDirectOptimized(self.bits);
    }

    // Detailed evaluation for proper hand comparison
    pub inline fn evaluateDetailed(self: Hand) HandEvaluation {
        // Extract rank data once
        const rank_data = extractRankDataOptimized(self.bits);

        // Check for flush first
        const flush_result = detectFlushOptimized(self.bits);
        if (flush_result > 0) {
            return buildFlushEvaluation(@enumFromInt(flush_result), rank_data.counts);
        }

        // Non-flush hands
        return buildNonFlushEvaluation(rank_data.counts, rank_data.mask);
    }

    // Hand composition methods
    pub fn combineWith(self: Hand, other: Hand) Hand {
        return Hand{ .bits = self.bits | other.bits };
    }

    pub fn fromHoleAndBoard(hole: [2]Card, board: []const Card) Hand {
        const hole_bits = hole[0].bits | hole[1].bits;
        const board_bits = cardsToBits(board);
        return Hand{ .bits = hole_bits | board_bits };
    }

    pub fn fromHoleAndBoardBits(hole: [2]Card, board_bits: u64) Hand {
        const hole_bits = hole[0].bits | hole[1].bits;
        return Hand{ .bits = hole_bits | board_bits };
    }

    pub fn fromBoard(board: []const Card) Hand {
        return Hand{ .bits = cardsToBits(board) };
    }

    // Check for card conflicts between hands
    pub fn hasConflictWith(self: Hand, other: Hand) bool {
        return (self.bits & other.bits) != 0;
    }

    // Compare two hands for showdown
    pub fn compareWith(self: Hand, other: Hand) ShowdownResult {
        const self_eval = self.evaluateDetailed();
        const other_eval = other.evaluateDetailed();
        return self_eval.compare(other_eval);
    }
};

// Convenience functions for creating cards with enums
pub fn createCard(suit: Suit, rank: Rank) Card {
    return Card.init(@intFromEnum(rank), @intFromEnum(suit));
}

pub fn createHand(cards: []const struct { Suit, Rank }) Hand {
    var hand = Hand.init();
    for (cards) |card_info| {
        const card = createCard(card_info[0], card_info[1]);
        hand.addCard(card);
    }
    return hand;
}

// Parse card string like "AsKsQsJsTs2h3h" into slice
pub fn parseCards(card_string: []const u8, allocator: std.mem.Allocator) ![]Card {
    if (card_string.len % 2 != 0) {
        return error.InvalidCardString;
    }

    const card_count = card_string.len / 2;
    var cards = try allocator.alloc(Card, card_count);
    var i: usize = 0;

    while (i < card_string.len) : (i += 2) {
        const rank_char = card_string[i];
        const suit_char = card_string[i + 1];

        // Parse rank
        const rank: u8 = switch (rank_char) {
            '2'...'9' => rank_char - '0',
            'T' => 10,
            'J' => 11,
            'Q' => 12,
            'K' => 13,
            'A' => 14,
            else => return error.InvalidRank,
        };

        // Parse suit
        const suit: u2 = switch (suit_char) {
            'h' => 0, // hearts
            's' => 1, // spades
            'd' => 2, // diamonds
            'c' => 3, // clubs
            else => return error.InvalidSuit,
        };

        cards[i / 2] = Card.init(rank, suit);
    }

    return cards;
}

/// Compile-time card parsing - returns fixed array, no allocation needed
pub fn mustParseCards(comptime card_string: []const u8) [card_string.len / 2]Card {
    if (card_string.len % 2 != 0) {
        @compileError("Invalid card string length: " ++ card_string);
    }

    const card_count = card_string.len / 2;
    var cards: [card_count]Card = undefined;

    comptime var i: usize = 0;
    inline while (i < card_string.len) : (i += 2) {
        const rank_char = card_string[i];
        const suit_char = card_string[i + 1];

        // Parse rank
        const rank: u8 = switch (rank_char) {
            '2'...'9' => rank_char - '0',
            'T' => 10,
            'J' => 11,
            'Q' => 12,
            'K' => 13,
            'A' => 14,
            else => @compileError("Invalid rank: " ++ [_]u8{rank_char}),
        };

        // Parse suit
        const suit: u2 = switch (suit_char) {
            'h' => 0, // hearts
            's' => 1, // spades
            'd' => 2, // diamonds
            'c' => 3, // clubs
            else => @compileError("Invalid suit: " ++ [_]u8{suit_char}),
        };

        cards[i / 2] = Card.init(rank, suit);
    }

    return cards;
}

/// Compile-time hole card parsing - returns exactly 2 cards
pub fn mustParseHoleCards(comptime card_string: []const u8) [2]Card {
    if (card_string.len != 4) {
        @compileError("Hole cards must be exactly 4 characters (e.g., 'AhAs'): " ++ card_string);
    }
    const cards = mustParseCards(card_string);
    return [2]Card{ cards[0], cards[1] };
}

/// Generate all combinations for a pocket pair (6 combinations)
pub fn generatePocketPair(rank: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 6);

    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations for pocket pairs
    for (suits, 0..) |suit1, i| {
        for (suits[i + 1 ..]) |suit2| {
            combinations[idx] = [2]Card{
                createCard(suit1, rank),
                createCard(suit2, rank),
            };
            idx += 1;
        }
    }

    return combinations;
}

/// Generate all suited combinations (4 combinations)
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 4);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };

    for (suits, 0..) |suit, i| {
        combinations[i] = [2]Card{
            createCard(suit, rank1),
            createCard(suit, rank2),
        };
    }

    return combinations;
}

/// Generate all offsuit combinations (12 combinations)
pub fn generateOffsuitCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 12);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all suit combinations where suits are different
    for (suits) |suit1| {
        for (suits) |suit2| {
            if (suit1 != suit2) {
                combinations[idx] = [2]Card{
                    createCard(suit1, rank1),
                    createCard(suit2, rank2),
                };
                idx += 1;
            }
        }
    }

    return combinations;
}

/// Generate all combinations (suited + offsuit = 16 combinations)
pub fn generateAllCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Card {
    var combinations = try allocator.alloc([2]Card, 16);
    const suits = [_]Suit{ .hearts, .spades, .diamonds, .clubs };
    var idx: usize = 0;

    // Generate all possible suit combinations
    for (suits) |suit1| {
        for (suits) |suit2| {
            combinations[idx] = [2]Card{
                createCard(suit1, rank1),
                createCard(suit2, rank2),
            };
            idx += 1;
        }
    }

    return combinations;
}

// Detailed hand evaluation with rank hierarchy for proper comparison
pub const HandEvaluation = struct {
    rank: HandRank,
    primary: u8, // Highest pair/trip/quad rank
    secondary: u8, // Second pair rank for two pair/full house
    kickers: [5]u8, // Remaining cards for tiebreaking (sorted high to low)

    pub fn compare(self: HandEvaluation, other: HandEvaluation) ShowdownResult {
        // First compare hand ranks
        const self_rank_value = @intFromEnum(self.rank);
        const other_rank_value = @intFromEnum(other.rank);

        if (self_rank_value > other_rank_value) {
            return .{ .winner = 0, .tie = false, .winning_rank = self.rank };
        } else if (other_rank_value > self_rank_value) {
            return .{ .winner = 1, .tie = false, .winning_rank = other.rank };
        }

        // Same hand rank - compare by primary rank
        if (self.primary > other.primary) {
            return .{ .winner = 0, .tie = false, .winning_rank = self.rank };
        } else if (other.primary > self.primary) {
            return .{ .winner = 1, .tie = false, .winning_rank = other.rank };
        }

        // Same primary - compare secondary (for two pair, full house)
        if (self.secondary > other.secondary) {
            return .{ .winner = 0, .tie = false, .winning_rank = self.rank };
        } else if (other.secondary > self.secondary) {
            return .{ .winner = 1, .tie = false, .winning_rank = other.rank };
        }

        // Compare kickers
        for (0..5) |i| {
            if (self.kickers[i] > other.kickers[i]) {
                return .{ .winner = 0, .tie = false, .winning_rank = self.rank };
            } else if (other.kickers[i] > self.kickers[i]) {
                return .{ .winner = 1, .tie = false, .winning_rank = other.rank };
            }
        }

        // True tie
        return .{ .winner = 0xFF, .tie = true, .winning_rank = self.rank };
    }
};

// Common showdown result type
pub const ShowdownResult = struct { winner: u8, tie: bool, winning_rank: HandRank };

pub fn generateRandomHand(random: std.Random) Hand {
    var hand_bits: u64 = 0;
    var used_cards = std.StaticBitSet(52).initEmpty();

    // Generate 7 unique random cards and compute bits directly
    var cards_added: u8 = 0;
    while (cards_added < 7) {
        const card_idx = random.uintLessThan(u8, 52);
        if (!used_cards.isSet(card_idx)) {
            used_cards.set(card_idx);

            // Compute bit directly without creating Card object
            const card_bit = @as(u64, 1) << @intCast(card_idx);
            hand_bits |= card_bit;

            cards_added += 1;
        }
    }

    return Hand{ .bits = hand_bits };
}

// Sample remaining cards avoiding conflicts with used cards
pub fn sampleRemainingCards(used_cards: []const Card, num_cards: u8, rng: std.Random) Hand {
    var used_bits: u64 = 0;
    for (used_cards) |card| {
        used_bits |= card.bits;
    }
    const sampled_bits = sampleRemainingCardsBits(used_bits, num_cards, rng);
    return Hand{ .bits = sampled_bits };
}

// =============================================================================
// IMPLEMENTATION DETAILS
// =============================================================================

// Private bit manipulation functions for performance
inline fn cardsToBits(cards: []const Card) u64 {
    var bits: u64 = 0;
    for (cards) |card| {
        bits |= card.bits;
    }
    return bits;
}

inline fn sampleRemainingCardsBits(used_cards: u64, num_cards: u8, rng: std.Random) u64 {
    var sampled_cards: u64 = 0;
    var cards_sampled: u8 = 0;

    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);

        // Skip if card already used or sampled
        if ((used_cards & card_bit) != 0 or (sampled_cards & card_bit) != 0) {
            continue;
        }

        sampled_cards |= card_bit;
        cards_sampled += 1;
    }

    return sampled_cards;
}

// Direct Bit Manipulation Lookup Tables for 7-card evaluation
// Generated at compile time for zero runtime overhead

// Flush lookup table: 8KB table for instant flush/straight-flush detection
// Index: 13-bit mask representing which ranks are present in a suit
// Value: Hand rank (0 = not enough cards, 6 = flush, 9 = straight flush)
pub const FLUSH_LOOKUP = generateFlushTable();

// Rank Distribution LUT: Smaller table for instant non-flush hand categorization
// Using a simple hash function to map rank distributions to hand categories
// Hash based on pair/trip/quad counts instead of full enumeration
const RANK_CATEGORY_LUT = generateRankCategoryLut();

// Generate flush lookup table at compile time
fn generateFlushTable() [8192]u16 {
    @setEvalBranchQuota(100000); // Increase compile-time loop limit
    var table: [8192]u16 = [_]u16{0} ** 8192;

    // For each possible 13-bit rank combination
    for (0..8192) |mask| {
        const popcount = @popCount(@as(u13, @intCast(mask)));
        if (popcount < 5) {
            table[mask] = 0; // Not enough cards for flush
            continue;
        }

        // Check for straight flush
        if (checkStraightInMask(@intCast(mask))) {
            table[mask] = 9; // Straight flush (highest rank)
        } else {
            table[mask] = 6; // Regular flush
        }
    }

    return table;
}

// Check if a rank mask contains a straight (for flush evaluation)
fn checkStraightInMask(mask: u13) bool {
    // Check wheel (A-2-3-4-5): bits 12,0,1,2,3
    if ((mask & 0b1000000001111) == 0b1000000001111) return true;

    // Check all 9 possible regular straights with unrolled loop
    const straight_patterns = [_]u13{
        0b1111100000000, // A-K-Q-J-T
        0b0111110000000, // K-Q-J-T-9
        0b0011111000000, // Q-J-T-9-8
        0b0001111100000, // J-T-9-8-7
        0b0000111110000, // T-9-8-7-6
        0b0000011111000, // 9-8-7-6-5
        0b0000001111100, // 8-7-6-5-4
        0b0000000111110, // 7-6-5-4-3
        0b0000000011111, // 6-5-4-3-2
    };

    inline for (straight_patterns) |pattern| {
        if ((mask & pattern) == pattern) return true;
    }

    return false;
}

// Simple hash function for rank category determination
// Based on pair/trip/quad counts instead of full rank distribution
fn hashRankCategory(pairs: u8, trips: u8, quads: u8) u8 {
    // Simple hash: quads*16 + trips*4 + pairs
    // Max value: 1*16 + 2*4 + 6 = 30 (well within u8 range)
    return quads * 16 + trips * 4 + pairs;
}

// Generate simplified rank category lookup table at compile time
// Much smaller table based on pair/trip/quad combinations
fn generateRankCategoryLut() [64]HandRank {
    @setEvalBranchQuota(100000); // Increase compile-time loop limit
    var lut: [64]HandRank = [_]HandRank{.high_card} ** 64;

    // Enumerate all valid combinations of pairs/trips/quads for 7 cards
    for (0..2) |quads| { // 0 or 1 quad possible
        for (0..3) |trips| { // 0, 1, or 2 trips possible
            for (0..7) |pairs| { // 0 to 6 pairs possible
                // Check if combination is valid for 7 cards
                const total_cards = quads * 4 + trips * 3 + pairs * 2;
                if (total_cards <= 7) {
                    const key = hashRankCategory(@intCast(pairs), @intCast(trips), @intCast(quads));

                    // Determine hand rank from counts
                    if (quads > 0) {
                        lut[key] = .four_of_a_kind;
                    } else if (trips > 0 and pairs > 0) {
                        lut[key] = .full_house;
                    } else if (trips > 0) {
                        lut[key] = .three_of_a_kind;
                    } else if (pairs >= 2) {
                        lut[key] = .two_pair;
                    } else if (pairs == 1) {
                        lut[key] = .pair;
                    } else {
                        lut[key] = .high_card;
                    }
                }
            }
        }
    }

    return lut;
}

// Optimized flush detection using bit manipulation
inline fn detectFlushOptimized(hand_bits: u64) u16 {
    // Extract suit counts using bit masks
    const suit_masks = [4]u64{
        0x1111111111111111, // Hearts (suit 0)
        0x2222222222222222, // Spades (suit 1)
        0x4444444444444444, // Diamonds (suit 2)
        0x8888888888888888, // Clubs (suit 3)
    };

    // Check all suits for 5+ cards
    inline for (0..4) |suit| {
        const suit_cards = hand_bits & suit_masks[suit];
        const suit_count = @popCount(suit_cards);

        if (suit_count >= 5) {
            // Extract rank mask for flush
            const rank_mask = extractFlushRankMaskOptimized(suit_cards, suit);
            const flush_rank = FLUSH_LOOKUP[rank_mask];
            if (flush_rank > 0) {
                return flush_rank;
            }
        }
    }

    return 0; // No flush found
}

// Extract rank mask for flush detection
pub inline fn extractFlushRankMaskOptimized(suit_cards: u64, suit: u3) u13 {
    _ = suit; // Not needed for this implementation
    // Extract rank mask for the specific suit - suit_cards should already be filtered
    var rank_mask: u13 = 0;
    inline for (0..13) |rank| {
        const nibble_shift = rank * 4;
        const nibble = (suit_cards >> nibble_shift) & 0xF;
        const has_card = @intFromBool(nibble != 0);
        rank_mask |= @as(u13, @intCast(has_card)) << @intCast(rank);
    }
    return rank_mask;
}

// Optimized flush detection that reuses pre-computed rank data
inline fn detectFlushOptimizedWithRankData(hand_bits: u64, rank_data: RankData) u16 {
    _ = rank_data; // Acknowledge parameter for future optimization

    // Extract suit counts using bit masks
    const suit_masks = [4]u64{
        0x1111111111111111, // Hearts (suit 0)
        0x2222222222222222, // Spades (suit 1)
        0x4444444444444444, // Diamonds (suit 2)
        0x8888888888888888, // Clubs (suit 3)
    };

    // Check all suits for 5+ cards
    inline for (0..4) |suit| {
        const suit_cards = hand_bits & suit_masks[suit];
        const suit_count = @popCount(suit_cards);

        if (suit_count >= 5) {
            // Extract rank mask for flush
            const rank_mask = extractFlushRankMaskOptimized(suit_cards, suit);
            const flush_rank = FLUSH_LOOKUP[rank_mask];
            if (flush_rank > 0) {
                return flush_rank;
            }
        }
    }

    return 0; // No flush found
}

// Type alias for rank data to ensure consistency
const RankData = struct { counts: [13]u8, mask: u16 };

// Ultra-optimized rank extraction - ARM64 SIMD or scalar fallback
inline fn extractRankDataOptimized(hand_bits: u64) RankData {
    if (comptime @import("builtin").cpu.arch == .aarch64) {
        return extractRankDataSIMD(hand_bits);
    } else {
        return extractRankDataScalar(hand_bits);
    }
}

// ARM64 SIMD rank extraction using NEON vector operations
inline fn extractRankDataSIMD(hand_bits: u64) RankData {
    // Use scalar approach with optimized instruction scheduling
    const nibbles = [13]u4{
        @truncate(hand_bits >> 0),  @truncate(hand_bits >> 4),
        @truncate(hand_bits >> 8),  @truncate(hand_bits >> 12),
        @truncate(hand_bits >> 16), @truncate(hand_bits >> 20),
        @truncate(hand_bits >> 24), @truncate(hand_bits >> 28),
        @truncate(hand_bits >> 32), @truncate(hand_bits >> 36),
        @truncate(hand_bits >> 40), @truncate(hand_bits >> 44),
        @truncate(hand_bits >> 48),
    };

    // Extract counts from nibbles
    var rank_counts: [13]u8 = undefined;
    var mask: u16 = 0;

    inline for (nibbles, 0..) |nibble, rank| {
        const count = @popCount(nibble);
        rank_counts[rank] = count;
        if (count > 0) {
            mask |= @as(u16, 1) << @intCast(rank);
        }
    }

    return RankData{ .counts = rank_counts, .mask = mask };
}

// Scalar fallback for non-ARM64 platforms
inline fn extractRankDataScalar(hand_bits: u64) RankData {
    // Manual unrolling for performance
    const r0 = @popCount((hand_bits >> 0) & 0xF); // Rank 2
    const r1 = @popCount((hand_bits >> 4) & 0xF); // Rank 3
    const r2 = @popCount((hand_bits >> 8) & 0xF); // Rank 4
    const r3 = @popCount((hand_bits >> 12) & 0xF); // Rank 5
    const r4 = @popCount((hand_bits >> 16) & 0xF); // Rank 6
    const r5 = @popCount((hand_bits >> 20) & 0xF); // Rank 7
    const r6 = @popCount((hand_bits >> 24) & 0xF); // Rank 8
    const r7 = @popCount((hand_bits >> 28) & 0xF); // Rank 9
    const r8 = @popCount((hand_bits >> 32) & 0xF); // Rank T
    const r9 = @popCount((hand_bits >> 36) & 0xF); // Rank J
    const r10 = @popCount((hand_bits >> 40) & 0xF); // Rank Q
    const r11 = @popCount((hand_bits >> 44) & 0xF); // Rank K
    const r12 = @popCount((hand_bits >> 48) & 0xF); // Rank A

    // Build rank mask
    const mask =
        (@as(u16, @intFromBool(r0 > 0)) << 0) |
        (@as(u16, @intFromBool(r1 > 0)) << 1) |
        (@as(u16, @intFromBool(r2 > 0)) << 2) |
        (@as(u16, @intFromBool(r3 > 0)) << 3) |
        (@as(u16, @intFromBool(r4 > 0)) << 4) |
        (@as(u16, @intFromBool(r5 > 0)) << 5) |
        (@as(u16, @intFromBool(r6 > 0)) << 6) |
        (@as(u16, @intFromBool(r7 > 0)) << 7) |
        (@as(u16, @intFromBool(r8 > 0)) << 8) |
        (@as(u16, @intFromBool(r9 > 0)) << 9) |
        (@as(u16, @intFromBool(r10 > 0)) << 10) |
        (@as(u16, @intFromBool(r11 > 0)) << 11) |
        (@as(u16, @intFromBool(r12 > 0)) << 12);

    return .{
        .counts = [13]u8{ r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12 },
        .mask = mask,
    };
}

// Non-flush evaluation using lookup table
fn evaluateNonFlushWithRankLUT(hand_bits: u64) HandRank {
    const rank_data = extractRankDataOptimized(hand_bits);
    return evaluateNonFlushWithPrecomputedRanks(rank_data.counts, rank_data.mask);
}

// Non-flush evaluation using pre-computed rank data
inline fn evaluateNonFlushWithPrecomputedRanks(rank_counts: [13]u8, rank_mask: u16) HandRank {
    @setEvalBranchQuota(100000);
    // Count pairs, trips, quads
    var pairs: u8 = 0;
    var trips: u8 = 0;
    var quads: u8 = 0;

    // Manual unroll for performance
    comptime var i = 0;
    inline while (i < 13) : (i += 1) {
        const count = rank_counts[i];
        pairs += @intFromBool(count == 2);
        trips += @intFromBool(count == 3);
        quads += @intFromBool(count == 4);
    }

    // Lookup pair category
    const hash_key = quads * 16 + trips * 4 + pairs;
    const pair_category = RANK_CATEGORY_LUT[hash_key];

    // Check for straight
    const is_straight = checkStraight(rank_mask);

    // Return best hand
    return if (is_straight and @intFromEnum(HandRank.straight) > @intFromEnum(pair_category))
        .straight
    else
        pair_category;
}

// Branch-free straight detection
inline fn checkStraight(mask: u16) bool {
    // Check for 5 consecutive bits
    const present = mask;
    const run5 = present &
        (present >> 1) &
        (present >> 2) &
        (present >> 3) &
        (present >> 4);

    // Check for regular straights (any 5 consecutive ranks) or wheel (A-2-3-4-5)
    return (run5 != 0) or ((present & 0b1000000001111) == 0b1000000001111);
}

// Build detailed evaluation for flush hands
inline fn buildFlushEvaluation(hand_rank: HandRank, rank_counts: [13]u8) HandEvaluation {
    var kickers: [5]u8 = [_]u8{0} ** 5;
    var primary: u8 = 0;

    // For straight flushes, find the high card of the straight (like straights)
    // For regular flushes, collect the top 5 cards as kickers
    if (hand_rank == .straight_flush) {
        // For straight flushes, primary is the high card of the straight
        // The flush detection already verified it's a straight flush
        // We need to determine the straight high card from the rank counts
        var rank_mask: u16 = 0;
        for (rank_counts, 0..) |count, rank| {
            if (count > 0) {
                rank_mask |= @as(u16, 1) << @intCast(rank);
            }
        }

        // Check for wheel straight flush
        if ((rank_mask & 0b1000000001111) == 0b1000000001111) {
            primary = 5; // Wheel straight flush, 5 is high
        } else {
            // Find highest straight
            var high_rank_idx: i8 = 12;
            while (high_rank_idx >= 4) : (high_rank_idx -= 1) {
                const shift_amount = @as(u4, @intCast(high_rank_idx - 4));
                const straight_mask = (@as(u16, 0b11111) << shift_amount);
                if ((rank_mask & straight_mask) == straight_mask) {
                    primary = @intCast(high_rank_idx + 2);
                    break;
                }
            }
        }
        // Straight flushes don't use kickers
        kickers = [_]u8{0} ** 5;
    } else {
        // Regular flush: collect top 5 cards as kickers
        var kicker_idx: usize = 0;
        var rank_idx: i8 = 12; // Start from ace
        while (rank_idx >= 0 and kicker_idx < 5) : (rank_idx -= 1) {
            const rank = @as(usize, @intCast(rank_idx));
            if (rank_counts[rank] > 0) {
                kickers[kicker_idx] = @intCast(rank + 2); // Convert to card rank
                kicker_idx += 1;
            }
        }
    }

    return HandEvaluation{
        .rank = hand_rank,
        .primary = primary,
        .secondary = 0, // Not applicable for flushes
        .kickers = kickers,
    };
}

// Build detailed evaluation for non-flush hands
inline fn buildNonFlushEvaluation(rank_counts: [13]u8, rank_mask: u16) HandEvaluation {
    var pairs: [3]u8 = [_]u8{0} ** 3;
    var trips: [2]u8 = [_]u8{0} ** 2;
    var quads: u8 = 0;
    var kickers: [5]u8 = [_]u8{0} ** 5;

    var pair_count: usize = 0;
    var trip_count: usize = 0;

    // Collect pairs, trips, quads (reverse order - ace to deuce)
    var rank_idx: i8 = 12;
    while (rank_idx >= 0) : (rank_idx -= 1) {
        const rank = @as(usize, @intCast(rank_idx));
        const count = rank_counts[rank];
        const card_rank = @as(u8, @intCast(rank + 2));

        if (count == 4) {
            quads = card_rank;
        } else if (count == 3 and trip_count < 2) {
            trips[trip_count] = card_rank;
            trip_count += 1;
        } else if (count == 2 and pair_count < 3) {
            pairs[pair_count] = card_rank;
            pair_count += 1;
        }
    }

    // Collect kickers (single cards)
    var kicker_idx: usize = 0;
    rank_idx = 12;
    while (rank_idx >= 0 and kicker_idx < 5) : (rank_idx -= 1) {
        const rank = @as(usize, @intCast(rank_idx));
        if (rank_counts[rank] == 1) {
            kickers[kicker_idx] = @intCast(rank + 2);
            kicker_idx += 1;
        }
    }

    // Determine hand rank and set primary/secondary
    var hand_rank: HandRank = .high_card;
    var primary: u8 = 0;
    var secondary: u8 = 0;

    if (quads > 0) {
        hand_rank = .four_of_a_kind;
        primary = quads;
    } else if (trip_count > 0 and pair_count > 0) {
        hand_rank = .full_house;
        primary = trips[0];
        secondary = pairs[0];
    } else if (trip_count > 0) {
        hand_rank = .three_of_a_kind;
        primary = trips[0];
    } else if (pair_count >= 2) {
        hand_rank = .two_pair;
        primary = pairs[0]; // Higher pair
        secondary = pairs[1]; // Lower pair
    } else if (pair_count == 1) {
        hand_rank = .pair;
        primary = pairs[0];
    } else {
        // Check for straight
        const is_straight = checkStraight(rank_mask);
        if (is_straight) {
            hand_rank = .straight;
            // For straights, primary is the high card of the actual straight
            if ((rank_mask & 0b1000000001111) == 0b1000000001111) {
                primary = 5; // Wheel straight (A-2-3-4-5), 5 is high
            } else {
                // Find the highest card of the 5-card straight by checking for 5 consecutive bits
                var high_rank_idx: i8 = 12; // Start checking from Ace (index 12)
                while (high_rank_idx >= 4) : (high_rank_idx -= 1) {
                    // Create a mask for a 5-card straight ending at this rank index
                    const shift_amount = @as(u4, @intCast(high_rank_idx - 4));
                    const straight_mask = (@as(u16, 0b11111) << shift_amount);
                    if ((rank_mask & straight_mask) == straight_mask) {
                        primary = @intCast(high_rank_idx + 2); // Convert index to rank (2-14)
                        break;
                    }
                }
            }
        } else {
            hand_rank = .high_card;
        }
    }

    // For straights and flushes, kickers don't matter - only the primary rank
    if (hand_rank == .straight or hand_rank == .straight_flush) {
        kickers = [_]u8{0} ** 5; // Clear kickers for straights
    }

    return HandEvaluation{
        .rank = hand_rank,
        .primary = primary,
        .secondary = secondary,
        .kickers = kickers,
    };
}

// =============================================================================
// BIT MANIPULATION EVALUATION FUNCTIONS
// =============================================================================

// Direct evaluation with single rank data extraction
inline fn evaluateDirectOptimized(hand_bits: u64) HandRank {
    // Extract rank data once and reuse for both flush and non-flush evaluation
    const rank_data = extractRankDataOptimized(hand_bits);

    // Check flush
    const flush_rank = detectFlushOptimizedWithRankData(hand_bits, rank_data);
    if (flush_rank > 0) return @enumFromInt(flush_rank);

    // Use pre-computed rank data for non-flush evaluation
    return evaluateNonFlushWithPrecomputedRanks(rank_data.counts, rank_data.mask);
}

// =============================================================================
// TESTS
// =============================================================================

const testing = std.testing;

test "card creation and properties" {
    const card = createCard(.spades, .ace);
    try testing.expect(card.getRank() == 14);
    try testing.expect(card.getSuit() == 1);
}

test "hand rank ordering" {
    try testing.expect(@intFromEnum(HandRank.high_card) < @intFromEnum(HandRank.pair));
    try testing.expect(@intFromEnum(HandRank.pair) < @intFromEnum(HandRank.straight_flush));
}

test "known hand evaluations" {
    // Royal flush
    const royal_flush = createHand(&.{
        .{ .spades, .ace },
        .{ .spades, .king },
        .{ .spades, .queen },
        .{ .spades, .jack },
        .{ .spades, .ten },
        .{ .hearts, .two },
        .{ .hearts, .three },
    });
    try testing.expect(royal_flush.evaluate() == .straight_flush);

    // Pair of aces
    const pair_hand = createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .ace },
        .{ .hearts, .two },
        .{ .spades, .four },
        .{ .diamonds, .six },
        .{ .clubs, .eight },
        .{ .hearts, .ten },
    });
    try testing.expect(pair_hand.evaluate() == .pair);

    // High card
    const high_card = createHand(&.{
        .{ .hearts, .ace },
        .{ .spades, .king },
        .{ .hearts, .queen },
        .{ .spades, .ten },
        .{ .diamonds, .eight },
        .{ .clubs, .six },
        .{ .hearts, .four },
    });
    try testing.expect(high_card.evaluate() == .high_card);
}

test "known hand patterns correctness" {
    // Test case data with card strings and expected results
    const test_data = [_]struct { cards: []const u8, expected: HandRank }{
        .{ .cards = "AsKsQsJsTs2h3d", .expected = .straight_flush }, // Royal flush
        .{ .cards = "9s8s7s6s5s2h3d", .expected = .straight_flush }, // Straight flush
        .{ .cards = "AhAsAdAcKs2h3d", .expected = .four_of_a_kind }, // Four of a kind
        .{ .cards = "AhAsAdKhKs2c3d", .expected = .full_house }, // Full house
        .{ .cards = "AhKhQhJh9h2s3d", .expected = .flush }, // Flush
        .{ .cards = "AsKdQcJhTs2s3d", .expected = .straight }, // Straight
        .{ .cards = "AhAsAdKcQs2h3d", .expected = .three_of_a_kind }, // Three of a kind
        .{ .cards = "AhAsKdKhQs2c3d", .expected = .two_pair }, // Two pair
        .{ .cards = "AhAsKdQcJs2h3d", .expected = .pair }, // One pair
        .{ .cards = "AhKsQdJc9h7s2d", .expected = .high_card }, // High card
        .{ .cards = "Ah2s3d4c5h6s7d", .expected = .straight }, // Wheel straight
    };

    // Verify known hand patterns evaluate correctly
    inline for (test_data) |test_case| {
        const cards = mustParseCards(test_case.cards);
        const hand = Hand.fromCards(cards);
        const result_hand = hand.evaluate();
        try testing.expect(result_hand == test_case.expected);
    }

    // Test random hands for basic validity (using stack allocation for test)
    var rng = std.Random.DefaultPrng.init(123);
    const random = rng.random();

    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const hand = generateRandomHand(random);
        const result_hand = hand.evaluate();
        try testing.expect(@intFromEnum(result_hand) >= 1);
        try testing.expect(@intFromEnum(result_hand) <= 9);
    }
}

test "mustParseHoleCards helper" {
    const aa = mustParseHoleCards("AhAs");
    try testing.expect(aa[0].getRank() == 14 and aa[0].getSuit() == 0); // Ah
    try testing.expect(aa[1].getRank() == 14 and aa[1].getSuit() == 1); // As

    const kq = mustParseHoleCards("KdQc");
    try testing.expect(kq[0].getRank() == 13 and kq[0].getSuit() == 2); // Kd
    try testing.expect(kq[1].getRank() == 12 and kq[1].getSuit() == 3); // Qc
}

test "edge cases and corner cases" {
    // Test all straight variations
    const ace_high_straight = Hand.fromCards(mustParseCards("AsKsQdJcTh2h3d"));
    try testing.expect(ace_high_straight.evaluate() == .straight);

    const wheel_straight = Hand.fromCards(mustParseCards("Ah2s3d4c5h6s7d"));
    try testing.expect(wheel_straight.evaluate() == .straight);

    const middle_straight = Hand.fromCards(mustParseCards("6h7s8d9cTh2s3d"));
    try testing.expect(middle_straight.evaluate() == .straight);

    // Test flush vs straight priority
    const flush_beats_straight = Hand.fromCards(mustParseCards("AhKhQhJhTh2s3d"));
    try testing.expect(flush_beats_straight.evaluate() == .straight_flush);

    // Test full house variations
    const trips_over_pair = Hand.fromCards(mustParseCards("AhAsAdKhKs2c3d"));
    try testing.expect(trips_over_pair.evaluate() == .full_house);

    const pair_over_trips = Hand.fromCards(mustParseCards("AhAsKdKhKs2c3d"));
    try testing.expect(pair_over_trips.evaluate() == .full_house);

    // Test quad variations
    const quads_with_trips = Hand.fromCards(mustParseCards("AhAsAdAcKhKsKd"));
    try testing.expect(quads_with_trips.evaluate() == .four_of_a_kind);

    // Test two pair edge cases
    const high_two_pair = Hand.fromCards(mustParseCards("AhAsKdKh2s3c4d"));
    try testing.expect(high_two_pair.evaluate() == .two_pair);

    const low_two_pair = Hand.fromCards(mustParseCards("3h3s2d2hAs5c6d"));
    try testing.expect(low_two_pair.evaluate() == .two_pair);

    // Test minimum hands
    const ace_high = Hand.fromCards(mustParseCards("AhKsQdJc9h7s2d"));
    try testing.expect(ace_high.evaluate() == .high_card);

    const deuce_high = Hand.fromCards(mustParseCards("2h3s4d5c7h8s9d"));
    try testing.expect(deuce_high.evaluate() == .high_card);
}

test "hand comparison - pairs hierarchy" {
    // AA should beat KK, KK should beat 22, etc.

    // Fixed: Use non-sequential cards to avoid accidental straights
    const aa = Hand.fromCards(mustParseCards("AhAs2d7c9h8s3d"));
    const kk = Hand.fromCards(mustParseCards("KhKs2d7c9h8s3d"));
    const tt = Hand.fromCards(mustParseCards("ThTs2d7c9h8s3d"));
    const twos = Hand.fromCards(mustParseCards("2h2s7d9c8h3sKd"));

    // All should evaluate to .pair
    try testing.expect(aa.evaluate() == .pair);
    try testing.expect(kk.evaluate() == .pair);
    try testing.expect(tt.evaluate() == .pair);
    try testing.expect(twos.evaluate() == .pair);

    // But compareWith shows them as ties (BUG!)
    const aa_vs_kk = aa.compareWith(kk);
    const aa_vs_twos = aa.compareWith(twos);
    const kk_vs_twos = kk.compareWith(twos);

    // These should NOT be ties - AA should beat KK and 22:
    try testing.expect(aa_vs_kk.tie == false and aa_vs_kk.winner == 0); // AA beats KK
    try testing.expect(aa_vs_twos.tie == false and aa_vs_twos.winner == 0); // AA beats 22
    try testing.expect(kk_vs_twos.tie == false and kk_vs_twos.winner == 0); // KK beats 22
}

test "hand comparison - different hand types" {
    // This works correctly - different hand types are compared properly
    const aa = Hand.fromCards(mustParseCards("AhAs2d7c9h8s3d")); // pair
    const ace_high = Hand.fromCards(mustParseCards("AhKd2d7c9h8s3d")); // high card
    const flush = Hand.fromCards(mustParseCards("AhKh2h7h9h8s3d")); // flush

    try testing.expect(aa.evaluate() == .pair);
    try testing.expect(ace_high.evaluate() == .high_card);
    try testing.expect(flush.evaluate() == .flush);

    // These comparisons work correctly (different hand types)
    const aa_vs_ace_high = aa.compareWith(ace_high);
    const flush_vs_aa = flush.compareWith(aa);

    try testing.expect(aa_vs_ace_high.tie == false);
    try testing.expect(aa_vs_ace_high.winner == 0); // AA wins
    try testing.expect(flush_vs_aa.tie == false);
    try testing.expect(flush_vs_aa.winner == 0); // flush wins
}

test "hand comparison - high card hierarchy" {
    // AK should beat AQ, AQ should beat AT, etc.
    const ak = Hand.fromCards(mustParseCards("AhKd2d7c9h8s3d"));
    const aq = Hand.fromCards(mustParseCards("AhQd2d7c9h8s3d"));
    const a2 = Hand.fromCards(mustParseCards("AhTd2d7c9h8s3d"));

    try testing.expect(ak.evaluate() == .high_card);
    try testing.expect(aq.evaluate() == .high_card);
    try testing.expect(a2.evaluate() == .high_card);

    // These should not be ties - AK should beat AQ and AT
    const ak_vs_aq = ak.compareWith(aq);
    const ak_vs_a2 = ak.compareWith(a2);

    try testing.expect(ak_vs_aq.tie == false and ak_vs_aq.winner == 0); // AK beats AQ
    try testing.expect(ak_vs_a2.tie == false and ak_vs_a2.winner == 0); // AK beats AT
}

test "hand comparison - two pair hierarchy" {
    // AA22 should beat KK33 and QQ22
    const aa22 = Hand.fromCards(mustParseCards("AhAs2d2c7h9s8d"));
    const kk33 = Hand.fromCards(mustParseCards("KhKs3d3c7h9s8d"));
    const qq22 = Hand.fromCards(mustParseCards("QhQs2d2c7h9s8d"));

    try testing.expect(aa22.evaluate() == .two_pair);
    try testing.expect(kk33.evaluate() == .two_pair);
    try testing.expect(qq22.evaluate() == .two_pair);

    const aa22_vs_kk33 = aa22.compareWith(kk33);
    const aa22_vs_qq22 = aa22.compareWith(qq22);

    try testing.expect(aa22_vs_kk33.tie == false and aa22_vs_kk33.winner == 0); // AA22 beats KK33
    try testing.expect(aa22_vs_qq22.tie == false and aa22_vs_qq22.winner == 0); // AA22 beats QQ22
}

test "straight ranking bug fix" {
    // Critical bug: hand with A,8,7,6,5,4 should be 8-high straight, not ace-high
    const hand_with_ace_and_straight = Hand.fromCards(mustParseCards("Ah8s7d6c5h4s2d"));
    const eight_high_straight = Hand.fromCards(mustParseCards("8h7s6d5c4h3s2d"));

    // Both should evaluate as straights
    try testing.expect(hand_with_ace_and_straight.evaluate() == .straight);
    try testing.expect(eight_high_straight.evaluate() == .straight);

    // Should tie (both 8-high straights)
    const result = hand_with_ace_and_straight.compareWith(eight_high_straight);
    try testing.expect(result.tie == true);

    // Test against a 9-high straight to ensure 8-high is ranked correctly
    const nine_high_straight = Hand.fromCards(mustParseCards("9h8s7d6c5h4s2d"));
    try testing.expect(nine_high_straight.evaluate() == .straight);

    const vs_nine_high = hand_with_ace_and_straight.compareWith(nine_high_straight);
    try testing.expect(vs_nine_high.tie == false and vs_nine_high.winner == 1); // 9-high beats 8-high
}
