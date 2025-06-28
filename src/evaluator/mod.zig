const std = @import("std");
const card = @import("card");

// Import internal modules
const evaluator_impl = @import("evaluator.zig");
const slow_evaluator = @import("slow_evaluator.zig");
const tables = @import("tables.zig");
const mphf = @import("mphf.zig");

// PUBLIC API - High-performance evaluation functions
pub const evaluateHand = evaluator_impl.evaluateHand;
pub const evaluateBatch4 = evaluator_impl.evaluateBatch4;
pub const evaluateBatchDynamic = evaluator_impl.evaluateBatchDynamic;

// Public types
pub const HandRank = u16;

// Hand categories (from weakest to strongest) - poker domain concept
pub const HandCategory = enum(u4) {
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

// Convert evaluator rank (lower=better) to HandCategory enum
pub fn getHandCategory(rank: HandRank) HandCategory {
    // Evaluator uses lower numbers for better hands
    // These ranges are based on the actual slow evaluator implementation
    if (rank <= 10) return .straight_flush; // Royal flush + straight flushes (0-10)
    if (rank <= 165) return .four_of_a_kind; // Four of a kind (10-165)
    if (rank <= 321) return .full_house; // Full house (166-321)
    if (rank <= 1598) return .flush; // Flush (322-1598)
    if (rank <= 1608) return .straight; // Straight (1599-1608)
    if (rank <= 2466) return .three_of_a_kind; // Three of a kind (1609-2466)
    if (rank <= 3324) return .two_pair; // Two pair (2467-3324)
    if (rank <= 6184) return .pair; // One pair (3325-6184)
    return .high_card; // High card (6185-7461)
}

// Utility functions for testing/debugging
pub const isFlushHand = evaluator_impl.isFlushHand;
pub const getFlushPattern = evaluator_impl.getFlushPattern;

// Build-time table generation is handled by standalone executable in build.zig

// Benchmarking functions
pub const benchmarkSingle = evaluator_impl.benchmarkSingle;
pub const benchmarkBatch = evaluator_impl.benchmarkBatch;

// Test utilities
pub const generateRandomHandBatch = evaluator_impl.generateRandomHandBatch;
pub const generateRandomHand = evaluator_impl.generateRandomHand;

// PUBLIC API - Card and Hand types (evaluator format)
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;
pub const makeCard = card.makeCard;
pub const makeCardFromEnums = card.makeCardFromEnums;
pub const parseCard = card.parseCard;
pub const parseCards = card.parseCards;
pub const mustParseCards = card.mustParseCards;
pub const makeHandFromCards = card.makeHandFromCards;
pub const makeHandFromHoleAndBoard = card.makeHandFromHoleAndBoard;
pub const hasCard = card.hasCard;
pub const countCards = card.countCards;
pub const getSuitMask = card.getSuitMask;
pub const generateSuitedCombinations = card.generateSuitedCombinations;
pub const generateOffsuitCombinations = card.generateOffsuitCombinations;
pub const generatePocketPair = card.generatePocketPair;

// Testing utilities (internal use)
pub const slow = struct {
    pub const evaluateHand = slow_evaluator.evaluateHand;
    pub const makeCard = slow_evaluator.makeCard;
    pub const Hand = slow_evaluator.Hand;
};

// Import tests (required for test discovery)
test {
    _ = evaluator_impl;
    _ = slow_evaluator;
    _ = @import("build_tables.zig");
    _ = card;
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
