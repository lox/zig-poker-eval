const std = @import("std");
const poker = @import("poker.zig");
const benchmark = @import("benchmark.zig");
const equity = @import("equity.zig");
const simulation = @import("simulation.zig");

// Detailed profiler for poker hand evaluation performance analysis
pub const Profiler = struct {
    const ProfileData = struct {
        name: []const u8,
        total_time_ns: u64,
        call_count: u64,
        min_time_ns: u64,
        max_time_ns: u64,
        
        pub fn averageTimeNs(self: ProfileData) f64 {
            return @as(f64, @floatFromInt(self.total_time_ns)) / @as(f64, @floatFromInt(self.call_count));
        }
    };
    
    data: std.ArrayList(ProfileData),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) Profiler {
        return Profiler{
            .data = std.ArrayList(ProfileData).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Profiler) void {
        self.data.deinit();
    }
    
    pub fn addSample(self: *Profiler, name: []const u8, time_ns: u64) !void {
        // Find existing entry or create new one
        for (self.data.items) |*entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                entry.total_time_ns += time_ns;
                entry.call_count += 1;
                entry.min_time_ns = @min(entry.min_time_ns, time_ns);
                entry.max_time_ns = @max(entry.max_time_ns, time_ns);
                return;
            }
        }
        
        // Create new entry
        try self.data.append(ProfileData{
            .name = name,
            .total_time_ns = time_ns,
            .call_count = 1,
            .min_time_ns = time_ns,
            .max_time_ns = time_ns,
        });
    }
    
    pub fn printResults(self: *Profiler) void {
        const print = std.debug.print;
        
        print("\n=== PERFORMANCE PROFILE RESULTS ===\n", .{});
        print("{s:<25} {s:>10} {s:>12} {s:>10} {s:>10} {s:>10}\n", .{
            "Function", "Calls", "Total (ms)", "Avg (ns)", "Min (ns)", "Max (ns)"
        });
        print("{s}\n", .{"-" ** 80});
        
        for (self.data.items) |entry| {
            const total_ms = @as(f64, @floatFromInt(entry.total_time_ns)) / 1_000_000.0;
            const avg_ns = entry.averageTimeNs();
            
            print("{s:<25} {d:>10} {d:>12.3} {d:>10.1} {d:>10} {d:>10}\n", .{
                entry.name,
                entry.call_count,
                total_ms,
                avg_ns,
                entry.min_time_ns,
                entry.max_time_ns,
            });
        }
    }
};

// High-resolution timer for micro-benchmarking
pub inline fn timeFunction(comptime func: anytype, args: anytype) u64 {
    const start = std.time.nanoTimestamp();
    _ = @call(.auto, func, args);
    const end = std.time.nanoTimestamp();
    return @intCast(end - start);
}

// Profile individual components of hand evaluation
pub fn profileHandEvaluation(allocator: std.mem.Allocator) !void {
    var profiler = Profiler.init(allocator);
    defer profiler.deinit();
    
    const print = std.debug.print;
    print("=== DETAILED HAND EVALUATION PROFILING ===\n", .{});
    
    // Generate test hands
    const hands = try poker.generateRandomHands(allocator, 100_000, 12345);
    defer allocator.free(hands);
    
    print("Profiling {} hands...\n", .{hands.len});
    
    // Profile full evaluation
    for (hands) |hand| {
        const time_ns = timeFunction(poker.Hand.evaluate, .{hand});
        try profiler.addSample("full_evaluation", time_ns);
    }
    
    // Profile rank extraction
    for (hands) |hand| {
        const time_ns = timeFunction(profileRankExtraction, .{hand.bits});
        try profiler.addSample("rank_extraction", time_ns);
    }
    
    // Profile flush detection
    for (hands) |hand| {
        const time_ns = timeFunction(profileFlushDetection, .{hand.bits});
        try profiler.addSample("flush_detection", time_ns);
    }
    
    // Profile straight detection on actual rank masks
    for (hands) |hand| {
        const rank_data = poker.extractRankDataOptimized(hand.bits);
        const time_ns = timeFunction(poker.checkStraight, .{rank_data.mask});
        try profiler.addSample("straight_detection", time_ns);
    }
    
    // Profile pair/trip/quad counting
    for (hands) |hand| {
        const rank_data = poker.extractRankDataOptimized(hand.bits);
        const time_ns = timeFunction(profilePairCounting, .{rank_data.counts});
        try profiler.addSample("pair_counting", time_ns);
    }
    
    profiler.printResults();
}

// Profile equity simulation components
pub fn profileEquitySimulation(allocator: std.mem.Allocator) !void {
    var profiler = Profiler.init(allocator);
    defer profiler.deinit();
    
    const print = std.debug.print;
    print("\n=== EQUITY SIMULATION PROFILING ===\n", .{});
    
    // Set up test scenario
    const hero_hole = [_]poker.Card{ poker.Card.init(14, 0), poker.Card.init(14, 1) }; // AA
    const villain_hole = [_]poker.Card{ poker.Card.init(13, 2), poker.Card.init(13, 3) }; // KK
    const board = [_]poker.Card{};
    
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    
    const hero_bits = simulation.cardsToHoleBits(hero_hole);
    const villain_bits = simulation.cardsToHoleBits(villain_hole);
    const board_bits = simulation.boardToBits(&board);
    const used_cards = hero_bits | villain_bits | board_bits;
    const cards_needed = 5;
    
    print("Profiling 10000 equity simulations...\n", .{});
    
    // Profile each component 10000 times
    for (0..10000) |_| {
        // 1. Sample remaining cards
        const time_sample = timeFunction(simulation.sampleRemainingCards, .{ used_cards, cards_needed, rng });
        try profiler.addSample("sample_remaining_cards", time_sample);
        
        // 2. Combine cards (using actual sampled cards)
        const remaining_board = simulation.sampleRemainingCards(used_cards, cards_needed, rng);
        const final_board = board_bits | remaining_board;
        
        const time_combine1 = timeFunction(simulation.combineCards, .{ hero_bits, final_board });
        try profiler.addSample("combine_hero_cards", time_combine1);
        
        const time_combine2 = timeFunction(simulation.combineCards, .{ villain_bits, final_board });
        try profiler.addSample("combine_villain_cards", time_combine2);
        
        // 3. Evaluate showdown
        const hero_hand = simulation.combineCards(hero_bits, final_board);
        const villain_hand = simulation.combineCards(villain_bits, final_board);
        const hands = [_]poker.Hand{ hero_hand, villain_hand };
        
        const time_showdown = timeFunction(profileShowdownWrapper, .{ &hands, allocator });
        try profiler.addSample("evaluate_showdown", time_showdown);
        
        // 4. Individual hand evaluations
        const time_hero_eval = timeFunction(poker.Hand.evaluate, .{hero_hand});
        try profiler.addSample("hero_hand_eval", time_hero_eval);
        
        const time_villain_eval = timeFunction(poker.Hand.evaluate, .{villain_hand});
        try profiler.addSample("villain_hand_eval", time_villain_eval);
    }
    
    profiler.printResults();
}

// Wrapper for profiling showdown evaluation
fn profileShowdownWrapper(hands: *const [2]poker.Hand, allocator: std.mem.Allocator) void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();
    
    const result = simulation.evaluateShowdown(hands, arena_allocator) catch return;
    _ = result; // Prevent optimization
}

// Helper functions for component profiling
fn profileRankExtraction(hand_bits: u64) void {
    _ = poker.extractRankDataOptimized(hand_bits);
}

fn profileFlushDetection(hand_bits: u64) bool {
    const suit_masks = [4]u64{
        0x1111111111111111, // Hearts
        0x2222222222222222, // Spades  
        0x4444444444444444, // Diamonds
        0x8888888888888888, // Clubs
    };
    
    inline for (0..4) |suit| {
        const suit_cards = hand_bits & suit_masks[suit];
        if (@popCount(suit_cards) >= 5) {
            return true;
        }
    }
    return false;
}

fn profilePairCounting(rank_counts: [13]u8) struct { pairs: u8, trips: u8, quads: u8 } {
    var pairs: u8 = 0;
    var trips: u8 = 0;
    var quads: u8 = 0;
    
    inline for (rank_counts) |count| {
        switch (count) {
            2 => pairs += 1,
            3 => trips += 1,
            4 => quads += 1,
            else => {},
        }
    }
    
    return .{ .pairs = pairs, .trips = trips, .quads = quads };
}

// Advanced profiling with instruction counting (Apple Silicon specific)
pub fn profileWithInstructionCounting(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("\n=== APPLE SILICON INSTRUCTION PROFILING ===\n", .{});
    
    // This would require linking with Apple's performance monitoring framework
    // For now, we'll use high-resolution timing as a proxy
    
    const hands = try poker.generateRandomHands(allocator, 10_000, 54321);
    defer allocator.free(hands);
    
    // Measure cycle counts by timing very short operations
    const iterations = 1_000_000;
    
    // Profile different rank extraction approaches
    print("Comparing rank extraction approaches ({} iterations)...\n", .{iterations});
    
    // Current optimized approach
    var dummy_result: u64 = 0;
    const start1 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        const rank_data = poker.extractRankDataOptimized(hand.bits);
        dummy_result += rank_data.mask;
    }
    const end1 = std.time.nanoTimestamp();
    
    // Original loop-based approach
    const start2 = std.time.nanoTimestamp();
    for (0..iterations) |i| {
        const hand = hands[i % hands.len];
        var rank_mask: u16 = 0;
        inline for (0..13) |rank_idx| {
            const rank_bits = (hand.bits >> (rank_idx * 4)) & 0xF;
            if (@popCount(rank_bits) > 0) {
                rank_mask |= @as(u16, 1) << @intCast(rank_idx);
            }
        }
        dummy_result += rank_mask;
    }
    const end2 = std.time.nanoTimestamp();
    
    const time1 = end1 - start1;
    const time2 = end2 - start2;
    
    print("Results:\n", .{});
    print("  Optimized approach: {d:.2}ns per call\n", .{@as(f64, @floatFromInt(time1)) / iterations});
    print("  Original approach:  {d:.2}ns per call\n", .{@as(f64, @floatFromInt(time2)) / iterations});
    print("  Speedup: {d:.2}x\n", .{@as(f64, @floatFromInt(time2)) / @as(f64, @floatFromInt(time1))});
    print("  Dummy result: {} (prevents optimization)\n", .{dummy_result});
}

// Memory access pattern analysis
pub fn profileMemoryAccess(allocator: std.mem.Allocator) !void {
    const print = std.debug.print;
    print("\n=== MEMORY ACCESS PATTERN ANALYSIS ===\n", .{});
    
    // Test cache behavior with different hand distributions
    const hands_sequential = try poker.generateRandomHands(allocator, 50_000, 1);
    defer allocator.free(hands_sequential);
    
    const hands_random = try poker.generateRandomHands(allocator, 50_000, 99999);
    defer allocator.free(hands_random);
    
    // Sequential access pattern (cache-friendly)
    print("Testing sequential access pattern...\n", .{});
    const start_seq = std.time.nanoTimestamp();
    var dummy_seq: u32 = 0;
    for (hands_sequential) |hand| {
        dummy_seq += @intFromEnum(hand.evaluate());
    }
    const end_seq = std.time.nanoTimestamp();
    
    // Random access pattern (cache-hostile)
    print("Testing random access pattern...\n", .{});
    const start_rand = std.time.nanoTimestamp();
    var dummy_rand: u32 = 0;
    for (hands_random) |hand| {
        dummy_rand += @intFromEnum(hand.evaluate());
    }
    const end_rand = std.time.nanoTimestamp();
    
    const time_seq = end_seq - start_seq;
    const time_rand = end_rand - start_rand;
    
    print("Results:\n", .{});
    print("  Sequential: {d:.2}M hands/sec\n", .{50_000.0 / (@as(f64, @floatFromInt(time_seq)) / 1_000_000_000.0) / 1_000_000.0});
    print("  Random:     {d:.2}M hands/sec\n", .{50_000.0 / (@as(f64, @floatFromInt(time_rand)) / 1_000_000_000.0) / 1_000_000.0});
    print("  Cache penalty: {d:.2}x slower\n", .{@as(f64, @floatFromInt(time_rand)) / @as(f64, @floatFromInt(time_seq))});
    print("  Checksums: seq={} rand={}\n", .{dummy_seq, dummy_rand});
}