const std = @import("std");
const poker = @import("poker.zig");
const simulation = @import("simulation.zig");

pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,
    
    pub fn winRate(self: EquityResult) f64 {
        return @as(f64, @floatFromInt(self.wins)) / @as(f64, @floatFromInt(self.total_simulations));
    }
    
    pub fn equity(self: EquityResult) f64 {
        const win_equity = @as(f64, @floatFromInt(self.wins));
        const tie_equity = @as(f64, @floatFromInt(self.ties)) * 0.5;
        return (win_equity + tie_equity) / @as(f64, @floatFromInt(self.total_simulations));
    }
};

// Head-to-head Monte Carlo equity calculation
pub fn equityMonteCarlo(
    hero_hole: [2]poker.Card,
    villain_hole: [2]poker.Card,
    board: []const poker.Card,
    simulations: u32,
    rng: std.Random,
    allocator: std.mem.Allocator
) !EquityResult {
    
    const hero_bits = simulation.cardsToHoleBits(hero_hole);
    const villain_bits = simulation.cardsToHoleBits(villain_hole);
    const board_bits = simulation.boardToBits(board);
    const used_cards = hero_bits | villain_bits | board_bits;
    
    const cards_needed = 5 - @as(u8, @intCast(board.len));
    
    var wins: u32 = 0;
    var ties: u32 = 0;
    
    for (0..simulations) |_| {
        // Sample remaining board cards
        const remaining_board = simulation.sampleRemainingCards(used_cards, cards_needed, rng);
        const final_board = board_bits | remaining_board;
        
        // Create final hands
        const hero_hand = simulation.combineCards(hero_bits, final_board);
        const villain_hand = simulation.combineCards(villain_bits, final_board);
        
        const hands = [_]poker.Hand{ hero_hand, villain_hand };
        const result = try simulation.evaluateShowdown(&hands, allocator);
        defer result.deinit(allocator);
        
        if (result.winners.len == 1) {
            if (result.winners[0] == 0) {
                wins += 1;
            }
        } else {
            ties += 1;
        }
    }
    
    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = simulations,
    };
}

// Head-to-head exact equity calculation
pub fn equityExact(
    hero_hole: [2]poker.Card,
    villain_hole: [2]poker.Card,
    board: []const poker.Card,
    allocator: std.mem.Allocator
) !EquityResult {
    
    const hero_bits = simulation.cardsToHoleBits(hero_hole);
    const villain_bits = simulation.cardsToHoleBits(villain_hole);
    const board_bits = simulation.boardToBits(board);
    const used_cards = hero_bits | villain_bits | board_bits;
    
    const cards_needed = 5 - @as(u8, @intCast(board.len));
    
    // Enumerate all possible board completions
    const combinations = try simulation.enumerateCardCombinations(used_cards, cards_needed, allocator);
    defer allocator.free(combinations);
    
    var wins: u32 = 0;
    var ties: u32 = 0;
    
    for (combinations) |remaining_board| {
        const final_board = board_bits | remaining_board;
        
        // Create final hands
        const hero_hand = simulation.combineCards(hero_bits, final_board);
        const villain_hand = simulation.combineCards(villain_bits, final_board);
        
        const hands = [_]poker.Hand{ hero_hand, villain_hand };
        const result = try simulation.evaluateShowdown(&hands, allocator);
        defer result.deinit(allocator);
        
        if (result.winners.len == 1) {
            if (result.winners[0] == 0) {
                wins += 1;
            }
        } else {
            ties += 1;
        }
    }
    
    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = @intCast(combinations.len),
    };
}

// Multi-way Monte Carlo equity calculation
pub fn equityMultiWayMonteCarlo(
    hands: [][2]poker.Card,
    board: []const poker.Card,
    simulations: u32,
    rng: std.Random,
    allocator: std.mem.Allocator
) ![]EquityResult {
    const num_players = hands.len;
    if (num_players < 2) return error.NotEnoughPlayers;
    
    // Convert to bitsets
    var hole_bits = try allocator.alloc(u64, num_players);
    defer allocator.free(hole_bits);
    
    var used_cards: u64 = simulation.boardToBits(board);
    for (hands, 0..) |hole, i| {
        hole_bits[i] = simulation.cardsToHoleBits(hole);
        used_cards |= hole_bits[i];
    }
    
    const cards_needed = 5 - @as(u8, @intCast(board.len));
    
    // Initialize results
    var results = try allocator.alloc(EquityResult, num_players);
    for (results) |*result| {
        result.* = EquityResult{ .wins = 0, .ties = 0, .total_simulations = simulations };
    }
    
    for (0..simulations) |_| {
        // Sample remaining board cards
        const remaining_board = simulation.sampleRemainingCards(used_cards, cards_needed, rng);
        const final_board = simulation.boardToBits(board) | remaining_board;
        
        // Create final hands
        var final_hands = try allocator.alloc(poker.Hand, num_players);
        defer allocator.free(final_hands);
        
        for (hole_bits, 0..) |hole, i| {
            final_hands[i] = simulation.combineCards(hole, final_board);
        }
        
        const result = try simulation.evaluateShowdown(final_hands, allocator);
        defer result.deinit(allocator);
        
        if (result.winners.len == 1) {
            results[result.winners[0]].wins += 1;
        } else {
            // Split pot - each winner gets a tie
            for (result.winners) |winner| {
                results[winner].ties += 1;
            }
        }
    }
    
    return results;
}

// Hero vs field Monte Carlo equity (returns only hero's equity)
pub fn heroVsFieldMonteCarlo(
    hero_hole: [2]poker.Card,
    villain_holes: [][2]poker.Card,
    board: []const poker.Card,
    simulations: u32,
    rng: std.Random,
    allocator: std.mem.Allocator
) !f64 {
    // Build hands array with hero first
    var all_hands = try allocator.alloc([2]poker.Card, villain_holes.len + 1);
    defer allocator.free(all_hands);
    
    all_hands[0] = hero_hole;
    @memcpy(all_hands[1..], villain_holes);
    
    const results = try equityMultiWayMonteCarlo(all_hands, board, simulations, rng, allocator);
    defer allocator.free(results);
    
    return results[0].equity();
}

// Range vs range Monte Carlo equity
pub fn rangeEquityMonteCarlo(
    hero_range: [][2]poker.Card,
    villain_range: [][2]poker.Card,
    board: []const poker.Card,
    simulations: u32,
    rng: std.Random,
    allocator: std.mem.Allocator
) !struct { hero_equity: f64, villain_equity: f64 } {
    var total_hero_equity: f64 = 0;
    var total_villain_equity: f64 = 0;
    var valid_combinations: u32 = 0;
    
    const board_bits = simulation.boardToBits(board);
    
    for (hero_range) |hero_hand| {
        for (villain_range) |villain_hand| {
            const hero_bits = simulation.cardsToHoleBits(hero_hand);
            const villain_bits = simulation.cardsToHoleBits(villain_hand);
            
            // Skip if hands conflict
            if ((hero_bits & villain_bits) != 0 or 
                (hero_bits & board_bits) != 0 or 
                (villain_bits & board_bits) != 0) {
                continue;
            }
            
            const result = try equityMonteCarlo(hero_hand, villain_hand, board, simulations, rng, allocator);
            total_hero_equity += result.equity();
            total_villain_equity += 1.0 - result.equity();
            valid_combinations += 1;
        }
    }
    
    if (valid_combinations == 0) {
        return error.NoValidCombinations;
    }
    
    return .{
        .hero_equity = total_hero_equity / @as(f64, @floatFromInt(valid_combinations)),
        .villain_equity = total_villain_equity / @as(f64, @floatFromInt(valid_combinations)),
    };
}

// Hand vs range Monte Carlo equity
pub fn handVsRangeMonteCarlo(
    hero_hole: [2]poker.Card,
    villain_range: [][2]poker.Card,
    board: []const poker.Card,
    simulations: u32,
    rng: std.Random,
    allocator: std.mem.Allocator
) !f64 {
    const hero_bits = simulation.cardsToHoleBits(hero_hole);
    const board_bits = simulation.boardToBits(board);
    
    var total_equity: f64 = 0;
    var valid_hands: u32 = 0;
    
    for (villain_range) |villain_hand| {
        const villain_bits = simulation.cardsToHoleBits(villain_hand);
        
        // Skip if hands conflict
        if ((hero_bits & villain_bits) != 0 or 
            (hero_bits & board_bits) != 0 or 
            (villain_bits & board_bits) != 0) {
            continue;
        }
        
        const result = try equityMonteCarlo(hero_hole, villain_hand, board, simulations, rng, allocator);
        total_equity += result.equity();
        valid_hands += 1;
    }
    
    if (valid_hands == 0) {
        return error.NoValidHands;
    }
    
    return total_equity / @as(f64, @floatFromInt(valid_hands));
}

// Tests
const testing = std.testing;

test "basic equity calculation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    
    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc
    
    const result = try equityMonteCarlo(aa, kk, &.{}, 50000, rng, allocator);
    
    // For now, just test that equity calculation works and is reasonable
    try testing.expect(result.equity() > 0.3);
    try testing.expect(result.equity() < 0.9);
    try testing.expect(result.total_simulations == 50000);
}

test "exact vs monte carlo equity" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    
    const aa = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AhAs
    const kk = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KdKc
    
    // Test on turn (only 1 card to come) - no conflicting cards
    const board_slice = [_]poker.Card{
        poker.Card.init(12, 0), // Qh
        poker.Card.init(11, 1), // Js  
        poker.Card.init(7, 2),  // 7d
        poker.Card.init(2, 3),  // 2c
    };
    
    const exact_result = try equityExact(aa, kk, &board_slice, allocator);
    const monte_carlo_result = try equityMonteCarlo(aa, kk, &board_slice, 10000, rng, allocator);
    
    // Results should be close (within 5%)
    const diff = @abs(exact_result.equity() - monte_carlo_result.equity());
    try testing.expect(diff < 0.05);
}
