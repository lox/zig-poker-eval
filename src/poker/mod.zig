const std = @import("std");

// Import internal modules
const poker_types = @import("poker.zig");
const equity_impl = @import("equity.zig");
const ranges_impl = @import("ranges.zig");
const notation_impl = @import("notation.zig");
const simulation_impl = @import("simulation.zig");

// PUBLIC API - Core poker types
pub const Suit = poker_types.Suit;
pub const Rank = poker_types.Rank;
pub const HandRank = poker_types.HandRank;
pub const Card = poker_types.Card;

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

// Import tests (required for test discovery)
test {
    _ = poker_types;
    _ = equity_impl;
    _ = ranges_impl;
    _ = notation_impl;
    _ = simulation_impl;
}