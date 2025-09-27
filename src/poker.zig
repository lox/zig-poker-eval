/// High-performance poker hand evaluator and equity calculator
///
/// This library provides:
/// - Fast 7-card hand evaluation (2-5ns per hand)
/// - Monte Carlo and exact equity calculations
/// - Range notation parsing ("AA,KK,AKs")
/// - SIMD-optimized batch processing
/// - Multi-threaded equity calculations
///
/// Basic usage:
/// ```zig
/// const poker = @import("poker");
/// const hole = poker.parseHand("AsKd");
/// const board = poker.parseHand("AhKsQc");
/// const final_hand = hole | board;
/// const rank = poker.evaluateHand(final_hand);
/// ```
const card = @import("card");
const evaluator = @import("evaluator");
const hand = @import("hand");
const range_mod = @import("range");
const equity = @import("equity");
const analysis = @import("analysis");
const draws = @import("draws");

// === CORE TYPES ===

/// Represents cards and hands as packed u64 bitfields
/// Each bit represents one card in the 52-card deck
/// Layout: [13 spades][13 hearts][13 diamonds][13 clubs]
pub const Hand = card.Hand;

/// Card suits (clubs=0, diamonds=1, hearts=2, spades=3)
pub const Suit = card.Suit;

/// Card ranks (two=0, three=1, ..., ace=12)
pub const Rank = card.Rank;

/// Hand evaluation rank (lower = stronger hand)
pub const HandRank = evaluator.HandRank;

/// Hand categories from high card to straight flush
pub const HandCategory = evaluator.HandCategory;

// === CARD CREATION AND PARSING ===

/// Create a single card using enum types for safety
/// Example: makeCard(.spades, .ace) creates Ace of Spades
pub const makeCard = card.makeCard;

/// Format a single card to poker notation
/// Example: formatCard(makeCard(.spades, .ace)) -> "As"
pub const formatCard = card.formatCard;

/// Parse a single card from string notation at compile time
/// Example: parseCard("As") -> Ace of Spades
pub const parseCard = card.parseCard;

/// Parse a single card from string notation at runtime
/// Example: try maybeParseCard("As") -> Ace of Spades
pub const maybeParseCard = card.maybeParseCard;

/// Compile-time parsing of any card string into a Hand (CardSet)
/// Example: parseHand("AsKdQh") -> Hand with 3 cards
/// Example: parseHand("AsKd") -> Hand with 2 cards (hole cards)
/// Example: parseHand("AsKdQhJsTs5h2d") -> Hand with 7 cards (full hand)
pub const parseHand = hand.parseHand;
pub const maybeParseHand = hand.maybeParseHand;

// === HAND OPERATIONS ===

/// Check if a hand contains a specific card
/// Example: hasCard(hand, .spades, .ace)
pub const hasCard = card.hasCard;

/// Count total cards in a hand
pub const countCards = card.countCards;

// === HAND EVALUATION ===

/// Evaluate a 7-card hand and return its rank
/// Lower ranks are stronger: 0=royal flush, 7461=worst high card
pub const evaluateHand = evaluator.evaluateHand;

/// Convert numeric rank to hand category enum
/// Example: getHandCategory(100) -> .four_of_a_kind
pub const getHandCategory = evaluator.getHandCategory;

/// Evaluate a batch of hands with configurable batch size
/// batchSize must be known at compile time for optimal performance
pub const evaluateBatch = evaluator.evaluateBatch;

/// Default batch size for optimal performance (32)
pub const DEFAULT_BATCH_SIZE = evaluator.DEFAULT_BATCH_SIZE;

/// Evaluate 32 hands simultaneously for optimal performance
/// Achieves ~4.1 ns/hand on modern CPUs
pub const evaluateBatch32 = evaluator.evaluateBatch32;

/// Cached board metadata for repeated showdown evaluations
pub const BoardContext = evaluator.BoardContext;

/// Initialize a board context from a board bitmask
pub const initBoardContext = evaluator.initBoardContext;

/// Evaluate hero/villain hole cards using a shared board context
pub const evaluateHoleWithContext = evaluator.evaluateHoleWithContext;

/// Evaluate a showdown using a shared board context
pub const evaluateShowdownWithContext = evaluator.evaluateShowdownWithContext;

/// Evaluate many showdowns sharing the same board context
pub const evaluateShowdownBatch = evaluator.evaluateShowdownBatch;

// === HAND COMBINATIONS ===

/// Generate all suited combinations for two ranks
/// Example: generateSuitedCombinations(.ace, .king, allocator) -> 4 hands
pub const generateSuitedCombinations = hand.generateSuitedCombinations;

/// Generate all offsuit combinations for two ranks
/// Example: generateOffsuitCombinations(.ace, .king, allocator) -> 12 hands
pub const generateOffsuitCombinations = hand.generateOffsuitCombinations;

/// Generate all combinations for a pocket pair
/// Example: generatePocketPair(.ace, allocator) -> 6 hands (all AA combos)
pub const generatePocketPair = hand.generatePocketPair;

// === RANGE NOTATION ===

/// Poker range type for storing hand combinations with probabilities
pub const Range = range_mod.Range;

/// Parse range notation into Range object
/// Example: parseRange("AA,KK,AKs,AKo", allocator)
pub const parseRange = range_mod.parseRange;

/// Common preflop ranges for quick setup
pub const CommonRanges = range_mod.CommonRanges;

// === EQUITY CALCULATIONS ===

/// Basic equity result with win/tie/loss statistics
pub const EquityResult = equity.EquityResult;

/// Detailed equity result with hand category breakdown
pub const DetailedEquityResult = equity.DetailedEquityResult;

/// Head-to-head Monte Carlo equity calculation
/// Example: monteCarlo([As,Ks], [Qd,Qh], [], 100000, rng, allocator)
pub const monteCarlo = equity.monteCarlo;

/// Detailed Monte Carlo with hand category tracking
pub const detailedMonteCarlo = equity.detailedMonteCarlo;

/// Exact equity calculation (enumerates all possibilities)
/// Example: exact([As,Ks], [Qd,Qh], [Ah,Kd,Qc,Jh], allocator)
pub const exact = equity.exact;

/// Multi-threaded Monte Carlo for high-volume calculations
/// Example: threaded([As,Ks], [Qd,Qh], [], 1000000, seed, allocator)
pub const threaded = equity.threaded;

/// Multi-way equity for tournament scenarios
/// Example: multiway([[As,Ks], [Qd,Qh], [Jc,Jd]], [], 50000, rng, allocator)
pub const multiway = equity.multiway;

/// Hero vs field equity calculation
/// Example: heroVsFieldMonteCarlo([As,Ks], [[Qd,Qh], [Jc,Jd]], [], 50000, rng, allocator)
pub const heroVsFieldMonteCarlo = equity.heroVsFieldMonteCarlo;

// === BENCHMARKING ===

/// Generate random 7-card hand for testing
pub const generateRandomHand = evaluator.generateRandomHand;

/// Generate batch of 4 random hands for SIMD testing
pub const generateRandomHandBatch = evaluator.generateRandomHandBatch;

/// Benchmark single-hand evaluation performance
pub const benchmarkSingle = evaluator.benchmarkSingle;

/// Benchmark batch evaluation performance
pub const benchmarkBatch = evaluator.benchmarkBatch;

// === UTILITY FUNCTIONS ===

/// Check if a hand qualifies as a flush (5+ cards of same suit)
pub const isFlushHand = evaluator.isFlushHand;

/// Extract flush pattern for flush evaluation
pub const getFlushPattern = evaluator.getFlushPattern;

/// Direct hand-vs-hand comparison for showdowns
pub const evaluateEquityShowdown = equity.evaluateEquityShowdown;

// === BOARD ANALYSIS ===

/// Board texture classification
pub const BoardTexture = analysis.BoardTexture;

/// Flush potential information
pub const FlushInfo = analysis.FlushInfo;

/// Straight potential information
pub const StraightInfo = analysis.StraightInfo;

/// Analyze how coordinated/dangerous a board is
pub const analyzeBoardTexture = analysis.analyzeBoardTexture;

/// Analyze flush potential on the board
pub const analyzeFlushPotential = analysis.analyzeFlushPotential;

/// Analyze straight potential on the board
pub const analyzeStraightPotential = analysis.analyzeStraightPotential;

/// Count pairs on the board
pub const countBoardPairs = analysis.countBoardPairs;

/// Check if board is monotone (all same suit)
pub const isMonotone = analysis.isMonotone;

/// Check if board is rainbow (all different suits)
pub const isRainbow = analysis.isRainbow;

/// Check if board has three or more to a flush
pub const hasFlushDraw = analysis.hasFlushDraw;

/// Check if board has three or more connected cards
pub const hasStraightDraw = analysis.hasStraightDraw;

/// Check if board is paired
pub const isPairedBoard = analysis.isPairedBoard;

// === DRAW DETECTION ===

/// Types of draws a hand can have
pub const DrawType = draws.DrawType;

/// Information about draws in a hand
pub const DrawInfo = draws.DrawInfo;

/// Detect all draws in a hand (hole cards + community cards)
pub const detectDraws = draws.detectDraws;

// === INTERNAL/TESTING UTILITIES ===

/// Slow evaluator for validation and testing (internal use)
/// Access slow evaluator functions for benchmarking validation
pub const slow = struct {
    /// Slow but accurate hand evaluation for validation
    pub const evaluateHand = evaluator.slow_evaluator.evaluateHand;
};

// Tests to ensure the public API works correctly
const std = @import("std");
const testing = std.testing;

test "public API basic usage" {
    // Test card creation and parsing
    const ace_spades = makeCard(.spades, .ace);
    const parsed_ace = parseCard("As");
    try testing.expect(ace_spades == parsed_ace);

    // Test hand evaluation
    const royal_flush = parseHand("AsKsQsJsTs5h2d");
    const rank = evaluateHand(royal_flush);
    const category = getHandCategory(rank);
    try testing.expect(category == .straight_flush);

    // Test hole cards
    const hole = parseHand("AsKd");
    try testing.expect(hasCard(hole, .spades, .ace));
    try testing.expect(hasCard(hole, .diamonds, .king));
    try testing.expect(countCards(hole) == 2);
}

test "equity calculation API" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    // Test basic equity calculation
    const aa = makeCard(.clubs, .ace) | makeCard(.diamonds, .ace); // Pocket aces
    const kk = makeCard(.hearts, .king) | makeCard(.spades, .king); // Pocket kings

    const result = try monteCarlo(aa, kk, &.{}, 10000, rng, allocator);

    // AA should have significant equity against KK
    try testing.expect(result.equity() > 0.75);
    try testing.expect(result.total_simulations == 10000);
}

test "range parsing API" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test range creation
    var test_range = try parseRange("AA,KK", allocator);
    defer test_range.deinit();

    // Should have 12 total combinations (6 for AA + 6 for KK)
    try testing.expect(test_range.handCount() == 12);
}

// Ensure all public API tests are discovered
test {
    std.testing.refAllDecls(@This());
}
