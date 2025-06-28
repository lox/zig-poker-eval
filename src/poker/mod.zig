const std = @import("std");
const evaluator = @import("evaluator");

// Import internal modules
const poker_types = @import("poker.zig");
const equity_impl = @import("equity.zig");
const ranges_impl = @import("ranges.zig");
const notation_impl = @import("notation.zig");
const simulation_impl = @import("simulation.zig");

// PUBLIC API - Core poker types (from evaluator - source of truth)
pub const Suit = evaluator.Suit;
pub const Rank = evaluator.Rank;
pub const Hand = evaluator.Hand;
pub const Card = evaluator.Hand; // Individual cards are also Hand (u64)

// Poker domain types
pub const HandRank = poker_types.HandRank;
pub const ShowdownResult = poker_types.ShowdownResult;

// Card and hand creation functions (from evaluator)
pub const makeCard = evaluator.makeCard;
pub const makeCardFromEnums = evaluator.makeCardFromEnums;
pub const parseCard = evaluator.parseCard;
pub const parseCards = evaluator.parseCards;
pub const mustParseCards = evaluator.mustParseCards;
pub const makeHandFromCards = evaluator.makeHandFromCards;
pub const makeHandFromHoleAndBoard = evaluator.makeHandFromHoleAndBoard;
pub const hasCard = evaluator.hasCard;
pub const countCards = evaluator.countCards;
pub const getSuitMask = evaluator.getSuitMask;

// Card generation functions (from evaluator)
pub const generateSuitedCombinations = evaluator.generateSuitedCombinations;
pub const generateOffsuitCombinations = evaluator.generateOffsuitCombinations;
pub const generatePocketPair = evaluator.generatePocketPair;

// PUBLIC API - Equity analysis
pub const EquityResult = equity_impl.EquityResult;
pub const DetailedEquityResult = equity_impl.DetailedEquityResult;
pub const monteCarlo = equity_impl.monteCarlo;
pub const detailedMonteCarlo = equity_impl.detailedMonteCarlo;
pub const exact = equity_impl.exact;

// PUBLIC API - Range analysis
pub const Range = ranges_impl.Range;
pub const parseRange = ranges_impl.parseRange;

// PUBLIC API - Notation parsing
pub const parse = notation_impl.parse;

// PUBLIC API - Monte Carlo simulation
pub const evaluateShowdown = simulation_impl.evaluateShowdown;
pub const evaluateShowdownHeadToHead = simulation_impl.evaluateShowdownHeadToHead;
pub const sampleRemainingCards = simulation_impl.sampleRemainingCards;

// Bridge functions for hand evaluation
pub fn evaluateHand(hand: Hand) HandRank {
    const rank = evaluator.evaluateHand(hand);
    return poker_types.convertEvaluatorRank(rank);
}

pub fn compareHands(hand1: Hand, hand2: Hand) ShowdownResult {
    const rank1 = evaluateHand(hand1);
    const rank2 = evaluateHand(hand2);

    if (@intFromEnum(rank1) > @intFromEnum(rank2)) {
        return .{ .winner = 0, .tie = false, .winning_rank = rank1 };
    } else if (@intFromEnum(rank2) > @intFromEnum(rank1)) {
        return .{ .winner = 1, .tie = false, .winning_rank = rank2 };
    } else {
        return .{ .winner = 0, .tie = true, .winning_rank = rank1 };
    }
}

// Convenience functions for poker notation
pub fn createCard(suit: Suit, rank: Rank) Card {
    return makeCardFromEnums(suit, rank);
}

pub fn createHand(cards: []const struct { Suit, Rank }) Hand {
    var hand: Hand = 0;
    for (cards) |card_info| {
        const card = createCard(card_info[0], card_info[1]);
        hand |= card;
    }
    return hand;
}

pub fn mustParseHoleCards(comptime card_string: []const u8) Hand {
    if (card_string.len != 4) {
        @compileError("Hole cards must be exactly 4 characters (e.g., 'AhAs'): " ++ card_string);
    }
    const cards = evaluator.mustParseCards(card_string);
    return cards[0] | cards[1];
}

// Import tests (required for test discovery)
test {
    _ = poker_types;
    _ = equity_impl;
    _ = ranges_impl;
    _ = notation_impl;
    _ = simulation_impl;
}

// Ensure all tests in this module are discovered
test {
    std.testing.refAllDecls(@This());
}
