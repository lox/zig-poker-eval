# EXPERIMENTS.md - Cutting-Edge Optimization Experiments

This document outlines research-backed optimization experiments to push the 7-card Texas Hold'em evaluator beyond the current 34M evaluations/second baseline.

## Current Performance Baseline

- **Performance**: 34M evaluations/second (~29ns per hand)
- **Estimated cycles**: ~110 cycles/hand on 3.8GHz core
- **Architecture**: Bit-packed cards in u64, inline loops, @popCount operations
- **Memory**: Cache-friendly, minimal allocations

## Priority 1: Low-Effort, High-Impact Optimizations

### 1.1 Enhanced Straight Detection with Lookup Table ❌ FAILED
**Expected Gain**: 3-5% (1-2 cycles saved)
**Actual Result**: 0.99x slower (0.78ns vs 0.77ns per call)
**Effort**: 0.5 days
**Complexity**: Low

**EXPERIMENT RESULT**: The lookup table approach was slower on Apple M1. The original shift-mask approach benefits from:
- Excellent branch prediction on M1
- Short loop optimization 
- Better instruction cache utilization

Replace the current shifting-mask straight detection with a 10-entry lookup table:
```zig
const STRAIGHT_PATTERNS = [_]u16{
    0b1111100000000,  // A-K-Q-J-T
    0b0111110000000,  // K-Q-J-T-9
    // ... 8 more patterns
    0b1000000001111,  // A-5-4-3-2 (wheel)
};

fn checkStraightLUT(mask: u16) bool {
    for (STRAIGHT_PATTERNS) |pattern| {
        if ((mask & pattern) == pattern) return true;
    }
    return false;
}
```

**Lessons Learned**: 
- Apple M1's branch predictor is excellent for short, predictable loops
- LUT doesn't always beat optimized loops on modern CPUs
- Need to measure each optimization on target hardware

### 1.2 Leading/Trailing Zero Count Optimizations
**Expected Gain**: 2-3% (1-2 cycles saved)
**Effort**: 0.5 days
**Complexity**: Low

Use `@clz` and `@ctz` for rank gap detection:
```zig
// Get highest/lowest ranks directly
const highest_rank = 14 - @clz(rank_mask);
const lowest_rank = 2 + @ctz(rank_mask);
```

### 1.3 Micro-architectural Tuning ✅ SUCCESS  
**Expected Gain**: 2-4% (1-3 cycles saved)
**Actual Result**: 12.4% gain (37.8M → 42.5M hands/sec, 26ns → 24ns)
**Effort**: 0.5 days
**Complexity**: Low

**EXPERIMENT RESULT**: Highly successful optimizations on Apple M1:

**Implemented optimizations:**
- Force `inline` on critical path functions (`evaluate`, `checkStraight`)
- Pre-computed suit bit masks instead of dynamic shifting
- Hoisted constants outside loops
- Optimized suit extraction using parallel bit masks

**Code changes:**
```zig
// Before: Dynamic suit extraction
inline for (0..4) |suit| {
    var suit_bits: u64 = 0;
    inline for (0..13) |rank| {
        const bit_pos = rank * 4 + suit;
        if ((self.bits >> bit_pos) & 1 != 0) {
            suit_bits |= @as(u64, 1) << rank;
        }
    }
    suit_counts[suit] = @popCount(suit_bits);
}

// After: Pre-computed masks + parallel extraction
const suit_masks = [4]u64{
    0x1111111111111111, // Hearts
    0x2222222222222222, // Spades  
    0x4444444444444444, // Diamonds
    0x8888888888888888, // Clubs
};
inline for (0..4) |suit| {
    const extracted = self.bits & suit_masks[suit];
    suit_counts[suit] = @popCount(extracted);
}
```

**Total Priority 1 Actual Gain**: 12.4% → 42.5M evaluations/second (exceeded expectations)

## Priority 2: Medium-Effort ARM64/Apple Silicon Optimizations

### 2.1 NEON SIMD Bit Operations
**Expected Gain**: 8-15% on Apple M1/M2
**Effort**: 2 days
**Complexity**: Medium

Leverage ARM64 NEON for parallel bit operations:
```zig
fn extractSuitMasksNEON(hand_bits: u64) [4]u8 {
    if (comptime builtin.cpu.arch == .aarch64) {
        // Use 128-bit NEON vectors for parallel suit extraction
        const vec_hand = @splat(4, hand_bits);
        const suit_masks = @Vector(4, u64){ 0x1111111111111111, 0x2222222222222222, 0x4444444444444444, 0x8888888888888888 };
        const extracted = vec_hand & suit_masks;
        return @bitCast([4]u8, @popCount(extracted));
    } else {
        return extractSuitsGeneric(hand_bits);
    }
}
```

### 2.2 Perfect Hash Two-Level Lookup
**Expected Gain**: 30-50% (could reach 55-60M eval/s)
**Effort**: 4 days
**Complexity**: Medium-High

Implement 2-level perfect hash reducing evaluation to 2 table lookups:
- Level 1: Hash rank pattern (7 choose 13 = 1716 entries)
- Level 2: Hash suit equivalence class
- Total memory: ~1.5MB (still L2 cache friendly)

```zig
const HASH_TABLE_L1: [1716]u16 = generateL1Table();
const HASH_TABLE_L2: [65536]HandRank = generateL2Table();

fn evaluateHash(hand: Hand) HandRank {
    const rank_pattern = extractRankPattern(hand.bits);
    const suit_class = extractSuitClass(hand.bits);
    const l1_idx = HASH_TABLE_L1[rank_pattern];
    return HASH_TABLE_L2[l1_idx | suit_class];
}
```

## Priority 3: High-Throughput Batch Processing

### 3.1 NEON Vectorized Batch Evaluator (Apple M1 Optimized)
**Expected Gain**: 3-4x throughput for batch processing
**Effort**: 4 days
**Complexity**: High

Implement ARM64 NEON vectorized evaluator optimized for Apple Silicon:
```zig
const LANES = if (builtin.cpu.arch == .aarch64) 2 else 1; // 128-bit NEON = 2x u64

fn evaluateBatchNEON(hands: []const Hand) []HandRank {
    const vec_hands = @Vector(2, u64);
    // Use NEON CNT instruction for parallel popcount
    // Use NEON bitwise ops for parallel mask operations
    return neonVectorizedEvaluate(vec_hands);
}
```

### 3.2 Prefetch and Memory Optimization
**Expected Gain**: 5-10% for batch workloads
**Effort**: 1 day
**Complexity**: Medium

Add prefetching for better memory pipeline utilization:
```zig
fn evaluateBatchPrefetch(hands: []const Hand) []HandRank {
    for (hands, 0..) |hand, i| {
        if (i + 8 < hands.len) {
            @prefetch(&hands[i + 8], .read);
        }
        results[i] = hand.evaluate();
    }
}
```

## Experimental Research Directions

### 4.1 NEON PMULL Straight Detection (ARM64)
**Expected Gain**: Uncertain, possibly neutral
**Effort**: 1 day
**Risk**: High complexity for uncertain benefit

Use ARM64 NEON PMULL for single-instruction straight detection:
```zig
fn checkStraightPMULL(mask: u16) bool {
    if (comptime builtin.cpu.arch == .aarch64) {
        // Use NEON polynomial multiply (equivalent to PCLMULQDQ)
        const result = @pmull(mask, STRAIGHT_CONSTANT);
        return (result & STRAIGHT_DETECT_BIT) != 0;
    } else {
        return checkStraightLUT(mask);
    }
}
```

### 4.2 Prime Product Fingerprinting
**Expected Gain**: Unknown, likely slower for 7-card
**Effort**: 1 week
**Risk**: High complexity, memory overhead

Alternative mathematical approach using prime factorization.

## Apple M1 Specific Optimizations

### Apple Silicon Advantages
- **Wide Execution**: 8 execution ports, excellent for bit manipulation
- **Large Caches**: 128KB L1D, 12MB L2 per cluster - perfect for lookup tables
- **High Memory Bandwidth**: 68GB/s unified memory ideal for batch processing
- **Branch Prediction**: Excellent predictor reduces branch costs in evaluation chains

### M1-Optimized Techniques

#### Exploit Wide Execution
```zig
// Parallel rank counting optimized for M1's wide execution
inline fn countRanksM1(hand_bits: u64) [13]u8 {
    // All 13 rank extractions can execute in parallel
    var counts: [13]u8 = undefined;
    inline for (0..13) |i| {
        counts[i] = @popCount((hand_bits >> (i * 4)) & 0xF);
    }
    return counts;
}
```

#### Cache-Resident Perfect Tables
```zig
// 64KB lookup table fits perfectly in M1's L1 cache
const M1_RANK_LUT: [16384]u8 = comptime generateRankLUT();
const M1_SUIT_LUT: [16384]u8 = comptime generateSuitLUT();
```

#### NEON-Optimized Batch Processing
```zig
// M1's NEON excels at 128-bit parallel operations
fn evaluateBatchM1(hands: []const Hand) []HandRank {
    // Process 2 hands per NEON vector
    // Leverage M1's excellent NEON throughput
    return batchEvaluateNEON128(hands);
}
```

## Implementation Strategy

### Phase 1: Quick Wins (Week 1)
1. Implement Priority 1 optimizations
2. Add cycle-accurate benchmarking
3. Measure per-component performance
4. Target: 37-38M eval/s

### Phase 2: Feature-Gated Optimizations (Week 2-3)
1. Implement BMI2 fast path with fallback
2. Prototype perfect hash lookup
3. Choose best approach based on profiling
4. Target: 45-60M eval/s

### Phase 3: Batch Processing (Week 4)
1. Implement SIMD batch evaluator
2. Add prefetching optimizations
3. Maintain separate APIs for latency vs throughput
4. Target: 4-8x batch throughput

## Benchmarking Framework

### Performance Metrics
```zig
const BenchmarkResult = struct {
    hands_per_second: f64,
    nanoseconds_per_hand: f64,
    cycles_per_hand: f64,
    cache_misses_per_1000: f64,
};

fn benchmarkEvaluator(evaluator: anytype, hands: []const Hand) BenchmarkResult {
    // Cycle-accurate timing using RDTSC
    // Cache performance via perf counters
    // Statistical significance testing
}
```

### Test Matrix
- **Hardware**: Apple M1/M2/M3 ARM64 (primary), Intel x86-64, AMD x86-64
- **Compilers**: Zig 0.12+ with `-target aarch64-macos -mcpu=apple_a14` optimization
- **Workloads**: Single evaluation, small batches (8-64), large batches (1000+)
- **Hand distributions**: Random, torture cases, real-world patterns
- **Apple Silicon specific**: Test with and without AMX coprocessor load

## Risk Mitigation

### Cross-Platform Compatibility
- Runtime CPU feature detection
- Graceful fallbacks for unsupported features
- Separate code paths with identical APIs

### Code Maintenance
- Keep scalar evaluator as reference implementation
- Comprehensive test suite covering all optimizations
- Performance regression detection in CI

### Memory Constraints
- Limit total lookup tables to <4MB
- Use `.rodata` section for shared memory
- Optional huge page support for batch processing

## Validation Strategy

### Correctness Verification
- Exhaustive testing against reference implementation
- Fuzzing with random hand generators
- Cross-validation with other poker libraries

### Performance Validation
- A/B testing of optimization combinations
- Statistical significance testing (p < 0.01)
- Performance regression detection

## Expected Timeline

| Phase | Duration | Target Performance (Apple M1) | Key Deliverables |
|-------|----------|-------------------------------|------------------|
| 1 | 1 week | 37-40M eval/s | ARM64 micro-optimizations |
| 2 | 2-3 weeks | 50-65M eval/s | NEON + lookup optimizations |
| 3 | 1 week | 3-4x batch throughput | NEON batch processor |

## Success Criteria (Apple M1 Targets)

- **Minimum**: 50% improvement (51M eval/s) over baseline
- **Target**: 90% improvement (65M eval/s) for single evaluations  
- **Stretch**: 300% improvement for NEON batch processing with maintained accuracy
- **Quality**: Zero correctness regressions, optimal ARM64 code generation