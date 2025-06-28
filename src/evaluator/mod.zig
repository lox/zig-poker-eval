const std = @import("std");

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

// Utility functions for testing/debugging
pub const isFlushHand = evaluator_impl.isFlushHand;
pub const getFlushPattern = evaluator_impl.getFlushPattern;

// Build-time table generation (for build.zig)
pub const generateTables = @import("build_tables.zig").main;

// Benchmarking functions
pub const benchmarkSingle = evaluator_impl.benchmarkSingle;
pub const benchmarkBatch = evaluator_impl.benchmarkBatch;

// Test utilities
pub const generateRandomHandBatch = evaluator_impl.generateRandomHandBatch;
pub const generateRandomHand = evaluator_impl.generateRandomHand;

// Testing utilities (internal use)
pub const slow = struct {
    pub const evaluateHand = slow_evaluator.evaluateHand;
    pub const makeCard = slow_evaluator.makeCard;
    pub const Hand = slow_evaluator.Hand;
};

// Import tests (required for test discovery)
test {
    _ = evaluator_impl;
}