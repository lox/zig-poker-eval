# Performance Optimization Experiments

**Current Baseline (HEAD)**: 3.27 ns/hand (306M hands/s)
- Single evaluation: ~3.3 ns/hand
- Batch evaluation (32 hands): 306M hands/second
- Showdown evaluation (batched): 13.4 ns/eval (board context + SIMD batch-32)
- Showdown evaluation (context): 15.4 ns/eval (board context + SIMD batch-2, Exp 20)

**Historical Progress**:
- Original baseline (pre-SIMD): 11.95 ns/hand
- After SIMD batching (Exp 6): 4.5 ns/hand (2.66× faster)
- After SIMD flush detection (Exp 12): 3.69 ns/hand (3.24× faster)
- After flush pattern table (Exp 13): 3.27 ns/hand ← **current**
- **Total improvement**: 3.65× from original baseline

**Showdown Progress**:
- Original (context-only path, scalar): 41.9 ns/eval (Exp 10 baseline)
- After batch-32 optimization (Exp 10): 13.4 ns/eval
- Context path optimized (Exp 20): 15.4 ns/eval ← **current single-pair**
- **Total improvement**: 2.7× faster (context path), 3.1× faster (batched)

---

*All benchmarks run on MacBook Air (M1, 2020), 16GB RAM. See [Test Environment](#test-environment) below or [performance.md](performance.md) for profiling setup and methodology.*

---

## Key Principles (TL;DR)

- **Algorithmic parallelism wins** - SIMD batching (Exp 6, 10, 12) delivered the biggest gains
- **Profile first, optimize second** - Theory-driven experiments (1-5, 11, 14-15) mostly failed; profile-guided (12-13) succeeded
- **Complexity tax is real** - Implementation overhead often exceeds theoretical gains
- **Respect the hardware** - Work with CPU strengths (SIMD width, prefetchers, cache hierarchy)
- **Measurement methodology matters** - Heavy instrumentation caused 3× overhead (Exp 9)

See [detailed learnings](#key-learnings) at the end for full analysis of what works and what doesn't.

---

## Experiment 1: Packed ARM64 Tables

**[Evaluator]** **❌ Failed** | +5.9% slower (12.65→11.95 ns/hand) | Complexity: High

**Approach**: Pack two 13-bit hand ranks into each u32 table entry to halve memory footprint on ARM64. Added architecture detection in build_tables.zig and dual table formats with bit manipulation for extraction.

**Implementation**:

- Modified CHD value table from `[CHD_TABLE_SIZE]u16` to `[CHD_TABLE_SIZE/2]u32`
- Added `unpack_rank_arm64()` function with bit shifts and masks
- Conditional table generation based on target architecture

**Why it failed**: Complexity overhead outweighed theoretical memory benefits. The 267KB working set already fits comfortably in L2 cache, so halving it provided no cache benefit. The bit manipulation added CPU cycles without reducing memory pressure.

---

## Experiment 2: Memory Prefetching

**[Evaluator]** **❌ Failed** | +9.4% slower (13.07→11.95 ns/hand) | Complexity: Medium

**Approach**: Add `@prefetch` instructions in batch evaluation loops to hint upcoming memory accesses to the CPU cache system.

**Implementation**:

```zig
@prefetch(&tables.chd_value_table[final_index], .{});
```

**Why it failed**: The 267KB table working set is cache-resident, not memory-bound. Prefetch hints added overhead without improving cache hit rates. The evaluator is compute-bound, not memory-bound.

---

## Experiment 3: Simplified Hash Function

**[Evaluator]** **❌ Failed** | +3.2% slower (12.33→11.95 ns/hand) | Complexity: Low

**Approach**: Replace 3-step Murmur-style hash with single multiply and shift for faster hash computation.

**Implementation**:

```zig
// Original: mix64 (3 operations)
result ^= result >> 33;
result *%= MAGIC_CONSTANT;
result ^= result >> 29;

// Simplified: fast_hash (1 operation)
return (x * 0x9e3779b97f4a7c15) >> 13;
```

**Why it failed**: The original 3-step hash has better distribution and instruction-level parallelism. Modern CPUs can pipeline the three operations effectively, while the simpler hash created more collisions in the CHD displacement array.

---

## Experiment 4: RPC Bit Manipulation

**[Evaluator]** **❌ Failed** | +7.4% slower (12.83→11.95 ns/hand) | Complexity: Medium

**Approach**: Replace nested loops (4 suits × 13 ranks = 52 iterations) with unrolled bit manipulation to count rank occurrences.

**Implementation**:

```zig
// Original: nested loops
for (0..4) |suit| {
    for (0..13) |rank| {
        if ((suit_mask & (1 << rank)) != 0) rank_counts[rank] += 1;
    }
}

// Optimized: unrolled with inline for
inline for (0..13) |rank| {
    if (suits[0] & rank_bit != 0) count += 1;
    if (suits[1] & rank_bit != 0) count += 1;
    // ... unrolled for all 4 suits
}
```

**Why it failed**: Zig's compiler already optimizes the nested loops effectively. Manual unrolling added code complexity and register pressure without improving the generated assembly. The simple nested loops have better instruction cache locality.

---

## Experiment 5: LUT-Based RPC Computation

**[Evaluator]** **❌ Failed** | +10.2% slower (14.10→12.79 ns/hand) | Complexity: High

**Approach**: Replace 52-iteration rank counting loop with O(1) lookup tables, based on o3-pro analysis and comprehensive profiling that identified RPC computation as consuming 98% of runtime.

**Implementation**:

```zig
// Original: 52-iteration nested loops
for (0..4) |suit| {
    for (0..13) |rank| {
        if ((suit_mask & (1 << rank)) != 0) rank_counts[rank] += 1;
    }
}

// LUT approach: O(1) bit manipulation
inline for (0..13) |rank| {
    const rank_bit = 1 << rank;
    const count = @popCount(
        (if (clubs & rank_bit != 0) 1 else 0) |
        (if (diamonds & rank_bit != 0) 1 else 0) << 1 |
        // ... for all suits
    );
    rpc = rpc * 5 + count;
}
```

**Research phase**: Generated 768-byte rank delta lookup tables, implemented comprehensive validation (100K hands), and used high-frequency profiling (3,886 samples) to confirm RPC computation bottleneck.

**Why it failed**: Despite being "O(1)" in theory, the bit manipulation approach had higher per-operation overhead than the simple nested loops. Zig's compiler already optimizes the original loops with excellent auto-vectorization and instruction scheduling. The complex bit operations caused register pressure and defeated compiler optimizations. This reinforced the core lesson: **simple code + compiler optimization often beats manual micro-optimizations**.

---

## Experiment 6: SIMD Batching (4-Hand Parallel)

**[Evaluator]** **✅ Success** | +60% faster (11.95→4.5 ns/hand) | Complexity: High

**Approach**: Process 4 poker hands simultaneously using true SIMD parallelism with ARM64 NEON via Zig @Vector types, rather than optimizing single-hand computation.

**Implementation**: Structure-of-arrays layout with SIMD vectors for each suit. Vectorized rank counting processes all hands simultaneously using ARM64 NEON instructions.

```zig
// Structure-of-arrays for SIMD processing
const Hands4 = struct {
    clubs: @Vector(4, u16),
    diamonds: @Vector(4, u16),
    hearts: @Vector(4, u16),
    spades: @Vector(4, u16),
};

// Vectorized rank counting - processes 4 hands in parallel
fn compute_rpc_simd4(hands4: Hands4) @Vector(4, u32) {
    // ... vectorized bit checks and base-5 encoding ...
}
```

**Benchmark Results**: 3.2M hands (100K batches of 32)

- **Single-hand**: 8.67 ns/hand (115.4 M hands/sec)
- **32-hand batch**: ~4.5 ns/hand (224M+ hands/sec)
- **Speedup**: 1.93x
- **Checksums**: Identical (correctness verified)

**Why it succeeded**: Instead of fighting compiler optimizations with single-hand complexity, leveraged algorithmic parallelism. ARM64 NEON SIMD units can genuinely process 4 values simultaneously, providing true throughput multiplication. The structure-of-arrays layout enables efficient vectorization of the rank counting bottleneck.

**Key insight**: All previous experiments failed because they added complexity to already-optimized single-hand code. This experiment succeeded by changing the **algorithm** from "optimize 1 hand" to "process 4 hands in parallel" - working with SIMD strengths rather than against compiler optimizations.

---

## Experiment 7: 8-Hand SIMD Batching

**[Evaluator]** **❌ Failed** | -16% slower (7.11→8.58 ns/hand) | Complexity: High

**Approach**: Scale successful 4-hand SIMD batching to 8-hand batching to further amortize fixed overhead costs across more hands per operation.

**Implementation**:

```zig
// 8-hand structure-of-arrays using 2×4-wide vectors
fn compute_rpc_simd8(hands: [8]u64) [8]u32 {
    // Use two 4-wide vectors for 8 hands
    const clubs_v0: @Vector(4, u16) = clubs[0..4].*;
    const clubs_v1: @Vector(4, u16) = clubs[4..8].*;
    // ... similar for diamonds, hearts, spades

    var rpc_vec0: @Vector(4, u32) = @splat(0);
    var rpc_vec1: @Vector(4, u32) = @splat(0);

    // Vectorized rank counting on both vector pairs
    inline for (0..13) |rank| {
        // Process both 4-hand groups in parallel
        const rank_count_vec0 = clubs_has0 + diamonds_has0 + hearts_has0 + spades_has0;
        const rank_count_vec1 = clubs_has1 + diamonds_has1 + hearts_has1 + spades_has1;

        rpc_vec0 = rpc_vec0 * five_vec + @as(@Vector(4, u32), rank_count_vec0);
        rpc_vec1 = rpc_vec1 * five_vec + @as(@Vector(4, u32), rank_count_vec1);
    }

    // Combine results from both vector groups
    return [8]u32{ result0[0], result0[1], result0[2], result0[3],
                   result1[0], result1[1], result1[2], result1[3] };
}
```

**Benchmark Results**: Per batch processing

- **32-hand batching**: ~4.5 ns/hand (224M+ hands/sec)
- **64-hand batching**: 4.31 ns/hand (232.0M hands/sec)
- **Performance**: Only 3.5% improvement, not worth the complexity

**Why it failed**: Register pressure and complexity overhead outweighed the theoretical benefits of larger batch sizes. M1's NEON architecture is optimized for 4-wide operations; doubling to 8-wide required managing twice as many vectors without proportional ALU resources. The additional vector management, memory layout complexity, and register spilling defeated the amortization benefits.

**Detailed analysis**:

- **Register pressure**: 8-hand processing requires ~16 SIMD registers vs 8 for 4-hand
- **Memory access patterns**: More complex structure-of-arrays layout hurt cache efficiency
- **Instruction cache pressure**: Larger unrolled loops increased I-cache misses
- **Flush fallback amplification**: 2.4% flush probability means 8-hand batches hit scalar fallback more often

**Key insight**: The "sweet spot" for SIMD batch size is architecture-dependent. ARM64 NEON's 4-wide design makes 4-hand batching optimal; scaling beyond this hits diminishing returns from resource constraints rather than algorithmic improvements.

---

## Experiment 8: Flush Fallback Elimination

**[Evaluator]** **✅ Success** | Neutral on mixed workload, significant on flush-heavy batches | Complexity: Low

**Approach**: Eliminate scalar fallback when batches contain mixed flush/non-flush hands by keeping SIMD pipeline active for non-flush hands and handling flush hands per-lane.

**Problem**: Previous implementation fell back to scalar evaluation for entire 4-hand batch if ANY hand was a flush (9.3% probability), wasting SIMD work on 3 non-flush hands.

**Implementation**:

```zig
// Detect flush hands but preserve SIMD pipeline
var flush_mask: u4 = 0;
for (hands_array, 0..) |hand, i| {
    if (is_flush_hand(hand)) {
        flush_mask |= (@as(u4, 1) << @intCast(i));
    }
}

// Always run SIMD path for RPC computation (amortizes cost)
const rpc_results = compute_rpc_simd4(hands_array);

// Per-lane result selection based on flush detection
inline for (0..4) |i| {
    if (flush_mask & (@as(u4, 1) << @intCast(i)) != 0) {
        // Flush hand - use flush lookup table
        const pattern = get_flush_pattern(hands_array[i]);
        results[i] = tables.flush_lookup_table[pattern];
    } else {
        // Non-flush hand - use CHD lookup with SIMD-computed RPC
        results[i] = chd_lookup_scalar(rpc_results[i]);
    }
}
```

**Benchmark Results**:

- **Mixed workload**: ~4.5 ns/hand (maintained performance)
- **Flush-heavy batch**: Minimal regression due to per-lane handling
- **Pure non-flush batch**: Optimal ~4.5 ns/hand performance

**Why it succeeded**: Eliminated 9.3% scalar penalty for mixed batches while maintaining SIMD efficiency. The approach respects ARM64 architecture constraints and avoids complexity tax by using simple per-lane selection rather than vectorized flush handling.

**Key insight**: Strategic fallback elimination can provide targeted improvements without adding architectural complexity. The SIMD pipeline investment is preserved even when some lanes require different processing paths.

---

## Experiment 9: Measurement Methodology Discovery (Reverted)

**[Evaluator]** **❌ Reverted** | Neutral - heavy measurement artifacts (3× overhead) | Complexity: High

**Critical discovery**: Heavy timing instrumentation caused **~22 ns/hand measurement overhead** (3x actual performance), completely misleading optimization efforts.

**Key lesson**: Always measure production performance without debugging/profiling overhead. "Heisenberg's uncertainty principle" applies to performance measurement.

---

## Experiment 10: Context-Aware Showdown Batching

**[Showdown]** **✅ Success** | 3.1× faster (41.9→13.4 ns/eval) | Complexity: Medium

**Approach**: Reuse board metadata and evaluate many hero/villain pairs at once. Instead of reassembling full 7-card hands per pair, precompute suit/rank counts with `BoardContext`, then pack up to 32 hero/villain pairs into vectors and call `evaluateBatch` for both sides, comparing ranks lane-by-lane.

**Implementation**:

- Added `BoardContext` helper exports (`initBoardContext`, `evaluateHoleWithContext`, `evaluateShowdownWithContext`) so callers can precompute board state once.
- Introduced `evaluateShowdownBatch` to process chunks of 32/16/8/4/2/1 pairs, combining the shared board mask with each hole mask and comparing SIMD rank vectors.
- Extended `poker-eval bench --showdown` to report context vs. batched timings and added regression tests ensuring batch results match the single context path.

**Benchmark** (`zig build -Doptimize=ReleaseFast`, Apple M1):

```
poker-eval bench --showdown --iterations 320000
Context path: 41.92 ns/eval
Batched path: 13.42 ns/eval
Speedup:      3.12×
```

**Why it worked**: BoardContext eliminates redundant suit/rank recomputation, and batching keeps the SIMD evaluator saturated. The cost of packing 32 pairs is amortized across the vector work, turning the showdown comparator into a memory-friendly, cache-resident loop with minimal scalar overhead.

---

## Experiment 11: Force Inline Hot Path Functions

**[Evaluator]** **❌ No improvement** | Neutral (4.35→4.38 ns/hand, within noise) | Complexity: Trivial

**Approach**: Add explicit `inline` keywords to frequently-called functions `isFlushHand()` and `chdLookupScalar()` to eliminate potential function call overhead in the batch evaluation loop.

**Hypothesis**: Based on deep analysis, scalar function call overhead for 32 calls/batch to these functions was theorized to represent 30-40% of evaluation time. Expected 20-35% speedup if compiler wasn't already inlining.

**Implementation**:

```zig
// src/evaluator.zig:298
inline fn chdLookupScalar(rpc: u32) u16 {
    return tables.lookup(rpc);
}

// src/evaluator.zig:304
pub inline fn isFlushHand(hand: u64) bool {
    const suits = [4]u16{
        @as(u16, @truncate(hand >> 0)) & RANK_MASK,
        @as(u16, @truncate(hand >> 13)) & RANK_MASK,
        @as(u16, @truncate(hand >> 26)) & RANK_MASK,
        @as(u16, @truncate(hand >> 39)) & RANK_MASK,
    };
    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) return true;
    }
    return false;
}
```

**Benchmark Results**: 100K iterations, ReleaseFast

- **Baseline**: 4.35 ns/hand (229.8M hands/sec)
- **With inline**: 4.37-4.38 ns/hand (228.1-228.8M hands/sec)
- **Change**: +0.7% slower (within measurement noise)
- **Tests**: All 78 tests passed

**Why it failed**: The Zig compiler (with LLVM backend) was **already automatically inlining** these functions at ReleaseFast optimization level. Adding explicit `inline` keywords provided no benefit because the optimizer had already identified these as hot-path functions worthy of inlining. The performance difference is within normal measurement variance.

**Key insight**: Modern optimizing compilers (LLVM) are highly effective at identifying hot paths and making inlining decisions. The performance profile was already optimal without manual intervention. This validates that the current 4.5ns/hand performance represents a genuine ceiling imposed by:
1. **Memory latency** for CHD table lookups (2 dependent loads per hand)
2. **Amdahl's Law** limiting further speedup when 30% of work is inherently scalar

**Conclusion**: The evaluator is **already operating near its theoretical performance limit** for the current algorithm. Further improvements require algorithmic changes (e.g., SIMD flush detection) rather than micro-optimizations that the compiler already handles.

---

## Experiment 12: SIMD Flush Detection

**[Evaluator]** **✅ Success** | +15.2% faster (4.35→3.69 ns/hand) | Complexity: Medium

**Approach**: Vectorize the 32 scalar `isFlushHand()` calls in the batch evaluation loop using SIMD @popCount operations to detect flush hands across the entire batch simultaneously.

**Motivation**: Profiling with uniprof (20M iterations) revealed that `isFlushHand()` accounted for 6-7% of execution time despite being inlined. The batch evaluation loop made 32 scalar calls per batch, each extracting suits and checking @popCount individually.

**Implementation**: Extract suits into structure-of-arrays, use vectorized `@popCount` to check all hands simultaneously, combine results with bitwise OR.

```zig
// SIMD flush detection - processes entire batch at once
fn detectFlushSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]bool {
    // Extract suits for all hands into arrays
    var clubs: [batchSize]u16 = undefined;
    // ... diamonds, hearts, spades ...

    // Vectorized popcount for all suits simultaneously
    const clubs_count = @popCount(@as(@Vector(batchSize, u16), clubs));
    // ... other suits ...

    const threshold: @Vector(batchSize, u16) = @splat(5);
    const has_flush = (clubs_count >= threshold) | (diamonds_count >= threshold) | ...;

    return has_flush; // ARM64 NEON CNT instruction
}
```

**Benchmark Results**: 100K iterations, ReleaseFast, Apple M1

- **Baseline (Experiment 11)**: 4.35 ns/hand (230M hands/sec)
- **With SIMD flush detection**: 3.69 ns/hand (271M hands/sec)
- **Improvement**: 15.2% faster (0.66 ns/hand reduction)
- **Consistency**: 3 runs: 3.66, 3.78, 3.62 ns/hand (CV: ~2%)
- **Tests**: All 78 tests passed

**Why it succeeded**: The profiling correctly identified flush detection as a 6-7% bottleneck. By extracting suits once for all 32 hands and using ARM64 NEON's vectorized @popCount instruction, we eliminated 32 scalar function calls and 128 individual @popCount operations (4 suits × 32 hands). The SIMD approach processes all hands' suit counts in parallel, achieving true data parallelism.

**Key insight**: Profile-guided optimization pays off. The uniprof profiling data (16,388 samples showing `isFlushHand` at line 313 consuming 6-7% of time) provided concrete evidence for this optimization target. The actual 15.2% improvement exceeded the predicted 6-7% because the SIMD approach also improved instruction cache utilization and reduced branch mispredictions.

**Technical details**:
- Uses ARM64 NEON `CNT` instruction for vectorized population count
- Bitwise OR (`|`) operator combines boolean vectors (Zig requirement)
- Supports batch sizes: 2, 4, 8, 16, 32, 64 (powers of 2)
- Fallback to scalar for other batch sizes

**Current performance ceiling**: At 3.69 ns/hand, we're approaching the theoretical limit:
- Memory latency (CHD lookups): ~2.0-2.5 ns/hand (unavoidable)
- Remaining computation: ~1.2-1.7 ns/hand
- Further improvements limited by Amdahl's Law with ~70% memory-bound work

---

## Experiment 13: Flush Pattern Lookup Table

**[Evaluator]** **✅ Success** | +11.4% faster (3.69→3.27 ns/hand) | Complexity: Medium

**Approach**: Replace branching and iteration in `getTop5Ranks()` with a compile-time lookup table that pre-computes the top-5 flush pattern for all 65,536 possible suit masks.

**Motivation**: Profiling with uniprof revealed that `getTop5Ranks()` consumed 5.26% of execution time despite being called only for flush hands (~20% of hands). The function had three code paths: exact-5-cards (fast path), straight detection (10 iterations), and rank iteration (up to 13 iterations). Analysis identified this as the largest non-memory bottleneck.

**Implementation**: Generate compile-time lookup table for all 65,536 possible suit masks. Pre-computes top-5 flush pattern handling exact-5-cards, straights, and high-card flushes.

```zig
// Compile-time lookup table - 128KB (65,536 × 2 bytes)
const flush_top5_table: [65536]u16 = blk: {
    @setEvalBranchQuota(2000000);
    var table: [65536]u16 = [_]u16{0} ** 65536;

    for (0..65536) |mask_int| {
        const suit_mask: u16 = @intCast(mask_int);
        // Handle: exact 5 cards, straights (10 patterns), high-card flush
        table[mask_int] = computeTop5Pattern(suit_mask);
    }

    break :blk table;
};

// Simplified function - single memory load replaces branching + iteration
inline fn getTop5Ranks(suit_mask: u16) u16 {
    return flush_top5_table[suit_mask];
}
```

**Benchmark Results**: 20M iterations (640M hands), ReleaseFast, Apple M1

- **Baseline (Experiment 12)**: 3.69 ns/hand (271M hands/sec)
- **With lookup table**: 3.27 ns/hand (306M hands/sec)
- **Improvement**: 11.4% faster (0.42 ns/hand reduction)
- **Consistency**: 3 runs: 3.26, 3.28, 3.28 ns/hand (CV: 0.4%)
- **Tests**: All 78 tests passed

**Why it succeeded**: The lookup table eliminates all branching and iteration in flush pattern extraction. Instead of checking for exact-5-cards, iterating through 10 straight patterns, then potentially iterating through 13 ranks, the function now performs a single array lookup. The 128KB table (65,536 × 2 bytes) is generated at compile-time and adds minimal cache pressure since the CHD tables (267KB) already force L2 cache usage.

**Key insight**: Profile-guided optimization continues to deliver results. The 5.26% profiling hotspot translated to an 11.4% improvement - the additional gain comes from:
1. Eliminating branch mispredictions (3 code paths → 1)
2. Reducing instruction cache pressure (complex loop logic → simple load)
3. Enabling better instruction pipelining (no data-dependent branching)

**Technical details**:
- Table size: 128KB (65,536 entries × 2 bytes per u16)
- Memory impact: Total working set now 395KB (267KB CHD + 128KB flush patterns)
- Cache behavior: Already L2-bound, minimal additional pressure
- Compile-time generation: Requires `@setEvalBranchQuota(2000000)` for table computation
- All possible suit masks pre-computed (valid inputs: 5-7 bits set)

**Performance analysis**:
- Current: 3.27 ns/hand (306M hands/sec)
- Theoretical ceiling: ~2.5 ns/hand (memory-bound CHD lookups)
- Remaining headroom: 0.77ns (23%)
- Further optimization limited by Amdahl's Law (~70% memory-bound work)

---

## Experiment 14: Reuse Suit Extraction

**[Evaluator]** **❌ Failed** | -1.8% slower (3.27→3.33 ns/hand) | Complexity: Medium

**Approach**: Eliminate redundant suit extraction by having `detectFlushSimd` return extracted suit data alongside the flush mask, allowing flush pattern evaluation to reuse this data instead of re-extracting suits in `getFlushPattern`.

**Motivation**: Analysis identified that suits are extracted twice per batch:
1. Once in `detectFlushSimd` for flush detection (~370-391 in evaluator.zig)
2. Again in `getFlushPattern` for each flush hand (~440-467)

For batch-32 with ~20% flush rate (~6 flush hands), this meant 24 redundant suit extractions (4 shifts + 4 masks each).

**Implementation**:

```zig
// New struct to return both flush mask and suit data
fn FlushDetectionResult(comptime batchSize: usize) type {
    return struct {
        flush_mask: [batchSize]bool,
        suits: [batchSize][4]u16, // Suits for reuse in flush evaluation
    };
}

// Modified detectFlushSimd to return suit data
fn detectFlushSimdWithSuits(comptime batchSize: usize, hands: *const [batchSize]u64) FlushDetectionResult(batchSize) {
    var result = FlushDetectionResult(batchSize){
        .flush_mask = [_]bool{false} ** batchSize,
        .suits = [_][4]u16{[_]u16{0} ** 4} ** batchSize,
    };

    // ... existing suit extraction logic ...

    // Store both results and suit data
    inline for (0..batchSize) |i| {
        result.flush_mask[i] = has_flush[i];
        result.suits[i] = [4]u16{ clubs[i], diamonds[i], hearts[i], spades[i] };
    }

    return result;
}

// New helper to get flush pattern from pre-extracted suits
inline fn getFlushPatternFromSuits(suits: [4]u16) u16 {
    for (suits) |suit_mask| {
        if (@popCount(suit_mask) >= 5) {
            return getTop5Ranks(suit_mask);
        }
    }
    return 0;
}

// Modified evaluateBatch to use pre-extracted suits
pub fn evaluateBatch(comptime batchSize: usize, hands: @Vector(batchSize, u64)) @Vector(batchSize, u16) {
    const flush_detection = detectFlushSimdWithSuits(batchSize, &hands_array);

    inline for (0..batchSize) |i| {
        if (flush_detection.flush_mask[i]) {
            const pattern = getFlushPatternFromSuits(flush_detection.suits[i]);
            results[i] = tables.flushLookup(pattern);
        } else {
            results[i] = chdLookupScalar(rpc_results[i]);
        }
    }
}
```

**Benchmark Results**: 20M+ iterations, ReleaseFast, Apple M1

- **Baseline (Experiment 13)**: 3.27 ns/hand (306M hands/sec)
- **With suit reuse**: 3.33 ns/hand (300M hands/sec)
- **Performance**: 1.8% slower (0.06 ns/hand regression)
- **Consistency**: 3 runs: 3.33, 3.33, 3.34 ns/hand
- **Tests**: All 78 tests passed

**Why it failed**: The overhead of allocating and returning a larger struct outweighed the benefit of eliminating redundant suit extraction. Root causes:

1. **Stack allocation overhead**: The `FlushDetectionResult` struct requires 256 bytes for batch-32 (32 hands × 4 suits × 2 bytes). This stack allocation and initialization cost exceeded the savings from eliminating suit extraction.

2. **Wasted work**: Suit data is stored for all 32 hands but only used for ~6 flush hands (~20% of batch). The remaining ~26 non-flush hands pay the storage cost without benefit.

3. **Register pressure**: Returning a larger struct likely caused register spilling or less efficient register allocation in the hot path.

4. **Memory bandwidth**: Storing 256 bytes per batch to save 24 suit extractions (96 bytes of shifts/masks) increased memory traffic rather than reducing it.

**Key insight**: **Premature data reuse can hurt performance**. While eliminating redundant computation sounds beneficial, the overhead of preserving and passing data structures can exceed the savings. The original design's "extract on demand" approach was actually more efficient because:
- Suit extraction for 6 flush hands (24 operations) is cheaper than allocating/storing/passing 256 bytes
- Compiler can optimize the redundant extraction better than managing the larger struct
- Cache pressure from larger stack frames outweighs computation savings

**Theoretical vs. actual**: The analysis correctly identified redundant work (suits extracted twice), but failed to account for:
- Cost of struct allocation and passing
- Efficiency of modern compilers at optimizing small, repeated operations
- Cache/memory hierarchy effects of larger data structures

**Conclusion**: At 3.27 ns/hand, we've reached a **practical performance ceiling**. The evaluator is 77% of theoretical maximum (2.5 ns/hand), with remaining time dominated by unavoidable CHD memory lookups. Further micro-optimizations face diminishing returns where implementation overhead exceeds theoretical savings.

---

## Experiment 15: Explicit CHD Prefetching

**[Evaluator]** **❌ Failed** | -16% slower (3.36→3.91 ns/hand) | Complexity: Medium

**Approach**: Add explicit software prefetch hints for CHD table lookups since all RPCs are computed before any lookups occur. The idea was to prefetch the final CHD value_table indices while the CPU processes earlier computations, hiding L2 cache latency.

**Motivation**: Analysis identified that CHD lookups dominate the 23% gap to theoretical maximum (2.5 ns/hand). The 256KB value_table doesn't fit in L1 cache (32KB), forcing L2 access with ~14-cycle latency. Since `evaluateBatch` computes all RPCs upfront via `computeRpcSimd`, we have the opportunity to prefetch all lookup addresses before the evaluation loop.

**Implementation**:

```zig
// After RPC computation and flush detection, add prefetch loop
inline for (0..batchSize) |i| {
    if (!flush_mask[i]) {
        // Inline CHD hash computation to find final index
        const rpc = rpc_results[i];
        const h = @as(u64, rpc) *% 0xA2C48F57A5F9B3D1; // CHD magic constant
        const bucket = @as(u32, @intCast((h ^ (h >> 29)) >> 51));
        const g_value = tables.chd_g_array[bucket];  // First load
        const base_index = @as(u32, @intCast((h ^ (h >> 29)) & 0x1FFFF));
        const final_index = (base_index + g_value) & 131071;
        @prefetch(&tables.chd_value_table[final_index], .{}); // Prefetch hint
    }
}

// Then evaluate with (hopefully) cached data
inline for (0..batchSize) |i| {
    if (flush_mask[i]) {
        results[i] = tables.flushLookup(getFlushPattern(hands_array[i]));
    } else {
        results[i] = chdLookupScalar(rpc_results[i]);  // Should hit cache
    }
}
```

**Benchmark Results**: 10M iterations, ReleaseFast, Apple M1

- **Baseline (Experiment 13)**: 3.36 ns/hand (298M hands/sec)
- **With explicit prefetching**: 3.91 ns/hand (256M hands/sec)
- **Performance**: 16% slower (0.55 ns/hand regression)
- **Consistency**: 3 runs: 3.91, 3.92, 4.00 ns/hand
- **Tests**: All 78 tests passed

**Why it failed**: Multiple compounding factors made explicit prefetching counterproductive:

1. **Duplicate work overhead**: The prefetch loop recomputes the full CHD hash (multiply, XOR, shifts) for all non-flush hands, then the evaluation loop computes it again via `chdLookupScalar`. This doubles the hash computation cost.

2. **Hardware prefetcher interference**: Modern CPU prefetchers (especially Apple M1) are already effective at detecting sequential access patterns in the batch evaluation. Explicit `@prefetch` hints likely confused or displaced the hardware prefetcher's predictions.

3. **Branch overhead**: The `if (!flush_mask[i])` check in the prefetch loop adds 32 branch instructions that weren't there before. Even though ~80% are predictable (non-flush), the branch overhead exceeds any prefetch benefit.

4. **Memory bandwidth**: The prefetch loop adds extra memory accesses to `chd_g_array` (8KB, L1-cached) for all hands. While individually cheap (~1-3 cycles), 32 extra L1 accesses add measurable overhead.

5. **Register pressure**: Inlining the hash computation in two places (prefetch and evaluation) increases register usage, potentially causing spills and hurting the SIMD operations.

**Key insight**: **Hardware prefetchers + simple code beats manual prefetching**. At this level of optimization (3.36 ns/hand), the CPU's hardware prefetcher is already doing near-optimal work. The sequential access pattern in the evaluation loop (iterate through rpc_results, call chdLookupScalar for each) is exactly what hardware prefetchers excel at predicting. Adding explicit prefetch hints:
- Duplicates work (redundant hash computations)
- Interferes with hardware prediction
- Adds instruction and branch overhead

**Alternative considered**: "Fused" prefetch where we compute the hash once and store indices, then prefetch, then use stored indices. Analysis showed this would require:
- 32 × 4 bytes = 128 bytes stack allocation for indices
- Same duplicate work problem (hash computed separately from lookup)
- Experiment 14 already showed struct allocation overhead exceeds computational savings

**Theoretical vs. actual**: The hypothesis that prefetching could hide L2 latency was sound in isolation, but failed to account for:
- Hardware prefetching already present
- Cost of computing what to prefetch
- Interference with existing CPU optimizations

**Conclusion**: Explicit prefetching is counterproductive at this performance level. The 23% gap to theoretical maximum (2.5 ns/hand) is likely **fundamental**:
- CHD value_table (256KB) doesn't fit in L1 (32KB) → unavoidable L2 latency
- Hardware prefetching is already maximally effective
- Remaining gap is memory hierarchy physics, not software optimization opportunity

We've definitively reached the **practical performance ceiling** for this algorithmic approach. Further gains require different algorithms (GPU, different hash structures) or accepting 77% of theoretical maximum.

---

## Key Learnings

### What Works

1. **Algorithmic parallelism wins** (Exp 6, 10, 12)
   - SIMD batching delivered 2.66× speedup by processing 4 hands simultaneously
   - True parallelism beats micro-optimizations every time
   - Structure-of-arrays layout enables efficient vectorization

2. **Profile-guided optimization delivers** (Exp 12, 13)
   - Uniprof identified `isFlushHand` consuming 6-7% → vectorization gave 15.2% improvement
   - `getTop5Ranks` hotspot (5.26%) → lookup table delivered 11.4% improvement
   - Profiling reveals non-obvious bottlenecks that theory misses

3. **Lookup tables eliminate branching** (Exp 13)
   - 128KB compile-time table eliminated 3 code paths and iteration
   - Single memory load beats complex branching logic
   - Cache-resident tables have minimal overhead

4. **Hardware prefetchers are smart** (Exp 15)
   - Modern CPUs detect sequential access patterns automatically
   - Simple code + hardware prefetching > manual prefetch hints
   - Let the hardware do what it's optimized for

### What Doesn't Work

1. **Memory optimizations when cache-resident** (Exp 1, 2, 15)
   - 267KB working set fits in L2 cache → no memory pressure
   - Prefetching and packing add overhead without benefit
   - Not memory-bound, so memory optimizations fail

2. **Fighting the compiler** (Exp 3, 4, 5, 11)
   - Zig/LLVM already optimizes nested loops, hashing, inlining
   - Manual "optimizations" add complexity without gains
   - Simple code + compiler optimization > manual micro-optimizations

3. **Premature data reuse** (Exp 14)
   - Struct allocation overhead (256 bytes) > redundant extraction cost (24 ops)
   - Compiler optimizes small repeated operations efficiently
   - Cache pressure from larger structures outweighs computational savings

4. **Scaling beyond architecture sweet spot** (Exp 7)
   - ARM64 NEON optimized for 4-wide operations
   - 8-wide batching hit register pressure and diminishing returns
   - Architecture constraints matter more than theoretical benefits

### Core Principles

- **Complexity tax is real**: Implementation overhead often exceeds theoretical gains
- **Measure everything**: "Obvious" optimizations frequently backfire
- **Respect the hardware**: Work with CPU strengths (SIMD width, prefetchers, cache hierarchy)
- **Profile first, optimize second**: Theory-driven experiments (1-5, 11, 14, 15) mostly failed; profile-guided (12, 13) succeeded
- **Measurement methodology matters**: Heavy instrumentation caused 3× overhead (Exp 9), completely misleading results

### Performance Ceiling

At **3.27 ns/hand (77% of theoretical maximum)**:
- CHD lookups (~2.0-2.5 ns) are memory-latency bound (L2 cache)
- Remaining ~0.8 ns is computation
- Further gains require algorithmic changes (different hash structures, GPU acceleration)
- Accepting 77% efficiency is often the right engineering tradeoff

---

## Test Environment

**Hardware**: MacBook Air (M1, 2020)
**CPU**: Apple M1 (8-core, 4 performance + 4 efficiency)
**Memory**: 16 GB unified memory
**OS**: macOS 14.2.1 (23C71)
**Compiler**: Zig 0.14.0
**Build flags**: `-Doptimize=ReleaseFast -Dcpu=native`

For benchmarking methodology, profiling setup, and optimization workflow, see [performance.md](performance.md).

*All experiments measured with 5 benchmark runs of 100K batches (400K hands) each*

---

## Proposed Experiments: Exact Equity Optimization

**Current Baseline**: `evaluateEquityShowdown()` bottleneck - 85.6% of runtime
- **Profiled workload**: AA vs KK preflop = 61,642,944 evaluations (36 matchups × 1.7M boards)
- **Runtime**: 8.67 seconds total, 994ms in `evaluateEquityShowdown` (85.6%)
- **Problem**: Called 61M times individually (~16 ns per call)
- **Proven solution exists**: Experiment 10 pattern (board context + batching = 3.1× faster)

---

## Experiment 16: Board Context Reuse for Exact Equity

**[Exact Equity]** **✅ Success** | 2.4× faster (0.24→0.10s per matchup) | Complexity: Low

**Approach**: Apply Experiment 10's proven pattern - precompute board metadata once per board instead of re-evaluating full 7-card hands for each matchup. Changed `exact()` and `exactDetailed()` to use `evaluator.initBoardContext()` and `evaluator.evaluateShowdownWithContext()` instead of creating full 7-card hands.

**Motivation**: Profile shows `evaluateEquityShowdown` consuming 85.6% of runtime (994ms / 1,161ms). The function is called 61M times, each time reconstructing complete 7-card hands and evaluating them from scratch. Board context reuse eliminates redundant suit/rank computation - the same pattern that gave Experiment 10 a 3.1× speedup.

**Implementation**:
```zig
// Before: Full 7-card hand evaluation
const hero_hand = hero_hole_cards | complete_board;
const villain_hand = villain_hole_cards | complete_board;
const result = evaluateEquityShowdown(hero_hand, villain_hand);

// After: Board context reuse
const ctx = evaluator.initBoardContext(complete_board);
const result = evaluator.evaluateShowdownWithContext(&ctx, hero_hole_cards, villain_hole_cards);
```

Applied to both `exact()` (line 401-418) and `exactDetailed()` (line 454-477) in equity.zig.

**Benchmark Results**: ReleaseFast, Apple M1

**Preflop (1,712,304 boards)**:
- **Single matchup (AhAs vs KdKc)**: 0.10s (100ms)
- **Per-board time**: ~58 ns/board
- **Throughput**: ~17M boards/second

**Flop (990 boards)**:
- **Time**: 0.003s (3ms)
- **Per-board time**: ~3 ns/board

**Turn (44 boards)**:
- **Time**: 0.006s (6ms)
- **Per-board time**: ~136 ns/board

**Estimated speedup** (based on single matchup):
- **Previous estimated time**: ~0.24s per matchup (from 8.67s / 36 matchups)
- **Current measured time**: 0.10s per matchup
- **Speedup**: **2.4× faster** (within expected 2-3× range)

**Why it succeeded**:
- Directly mirrors Experiment 10's proven pattern (3.1× speedup for showdown batching)
- Eliminates redundant board suit/rank computation for each matchup
- Board context (`initBoardContext`) precomputes board state once, reused for both hero and villain
- Zero complexity tax - simple algorithmic change with no additional overhead
- Already validated in production code (`exactVsRandom` at equity.zig:552-574)

**Key insight**: Board context reuse is the foundational optimization for exact equity. The pattern applies universally: precompute shared board state once, evaluate multiple matchups against it. This sets the stage for Experiment 17 (SIMD batching) and Experiment 18 (hybrid approach) which can build on this foundation for multiplicative gains.

---

## Experiment 17: SIMD Batching for Board Evaluation

**[Exact Equity]** **✅ Success** | 5.9× faster over Exp 16 (0.10s→0.017s), 14.1× total | Complexity: Medium

**Approach**: Process boards in batches of 32, using SIMD to evaluate multiple boards simultaneously. Batch create 32 full 7-card hands for hero and villain, then use `evaluateBatch(32, ...)` to evaluate all hands in parallel.

**Motivation**: After Experiment 16 reduces per-board overhead, the next bottleneck becomes function call overhead for 1.7M boards. Experiment 6 proved SIMD batching delivers 2.66× speedup by processing multiple items simultaneously. Apply the same pattern to board evaluation.

**Implementation**:
```zig
pub fn exact(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    // ... existing setup ...

    const BATCH_SIZE = 32;
    var hero_batch: [BATCH_SIZE]Hand = undefined;
    var villain_batch: [BATCH_SIZE]Hand = undefined;

    var wins: u32 = 0;
    var ties: u32 = 0;
    var i: usize = 0;

    // Process boards in batches of 32
    while (i < board_completions.len) {
        const batch_size = @min(BATCH_SIZE, board_completions.len - i);

        // Prepare batch of hands
        for (0..batch_size) |j| {
            const complete_board = board_hand | board_completions[i + j];
            hero_batch[j] = hero_hole_cards | complete_board;
            villain_batch[j] = villain_hole_cards | complete_board;
        }

        // SIMD batch evaluation (existing optimized code path)
        const hero_ranks = evaluator.evaluateBatch(batch_size, hero_batch[0..batch_size].*);
        const villain_ranks = evaluator.evaluateBatch(batch_size, villain_batch[0..batch_size].*);

        // Compare results
        for (0..batch_size) |j| {
            if (hero_ranks[j] < villain_ranks[j]) {
                wins += 1;
            } else if (hero_ranks[j] == villain_ranks[j]) {
                ties += 1;
            }
        }

        i += batch_size;
    }

    return EquityResult{ .wins = wins, .ties = ties, .total_simulations = @intCast(board_completions.len) };
}
```

**Benchmark Results**: ReleaseFast, Apple M1

**Preflop (1,712,304 boards)**:
- **Time**: 0.017s (17ms)
- **Speedup over Exp 16**: 5.9× faster (100ms → 17ms)
- **Total speedup**: 14.1× faster than pre-Exp16 baseline
- **Throughput**: ~100M boards/second
- **Per-board time**: ~10 ns/board (down from ~58 ns in Exp 16)

**Other scenarios**:
- Flop (990 boards): <1ms (sub-millisecond)
- Turn (44 boards): <1ms (sub-millisecond)

**Why it succeeded**:
- SIMD batching processes 32 boards in parallel, amortizing overhead across all evaluations
- Leverages existing highly-optimized `evaluateBatch` (Experiment 6's 2.66× SIMD gains)
- Batch creation is cache-friendly with sequential memory access
- ARM64 NEON executes 4-wide operations; 32-batch = 8 SIMD passes per side
- Each batch does 64 evaluations (32 hero + 32 villain) with minimal overhead

**Key insight**: SIMD batching scales exceptionally well for exact equity because board enumeration is embarrassingly parallel. Unlike Experiment 16's board context (which optimizes per-board work), SIMD batching achieves true data parallelism across multiple boards. The 5.9× improvement (vs expected 2×) suggests the combination of reduced overhead + SIMD efficiency + cache locality creates multiplicative gains.

**Trade-off**: Loses Experiment 16's board context optimization for batched boards (creates full 7-card hands instead), but the SIMD parallelism more than compensates. Remainder boards use board context as fallback.

---

## Experiment 18: Hybrid Board Context + SIMD Batching

**[Exact Equity]** Expected: 8-12× faster | Complexity: High | Status: Proposed

**Approach**: Combine Experiments 16 and 17 - batch process boards with board context reuse. This is the exact pattern used in `exactVsRandom` (equity.zig:496-602), which is the most optimized equity calculation in the codebase.

**Motivation**: The `exactVsRandom` function already demonstrates the optimal pattern: create board context once per board, then batch-evaluate multiple matchups. For `exact()` we flip this: batch-process multiple boards, using board context for each. This hybrid approach should deliver multiplicative gains from both optimizations.

**Implementation** (Per-board batching approach):
```zig
pub fn exact(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    // ... existing validation and board enumeration ...

    var wins: u32 = 0;
    var ties: u32 = 0;

    for (board_completions) |board_completion| {
        const complete_board = board_hand | board_completion;

        // Create board context once for this board
        const ctx = evaluator.initBoardContext(complete_board);

        // Evaluate showdown with board context (no redundant board computation)
        const result = evaluator.evaluateShowdownWithContext(&ctx, hero_hole_cards, villain_hole_cards);

        if (result > 0) {
            wins += 1;
        } else if (result == 0) {
            ties += 1;
        }
    }

    return EquityResult{ .wins = wins, .ties = ties, .total_simulations = @intCast(board_completions.len) };
}
```

**Alternative** (Batch-board approach with context):
```zig
// For range vs range (future optimization), batch boards AND matchups:
const BATCH_SIZE = 32;
for (boards in batches of BATCH_SIZE) {
    for (board in batch) {
        const ctx = evaluator.initBoardContext(board);
        // Batch evaluate all matchups for this board using evaluateShowdownBatch
    }
}
```

**Why it should work**:
- Directly mirrors the proven `exactVsRandom` implementation (equity.zig:552-554)
- Combines board context reuse (Exp 10: 3.1× faster) with SIMD benefits
- Already validated in production code with excellent performance
- Eliminates both redundant board computation AND function call overhead

**Expected benchmark** (AA vs KK preflop):
- **Baseline**: 8.67 seconds (61.6M evaluations)
- **With hybrid approach**: 0.7-1.1 seconds (8-12× faster)
- **Per-evaluation**: ~16 ns/eval → ~1.5-2 ns/eval

**Key insight**: This mirrors `exactVsRandom`'s design. That function processes 2B evaluations efficiently by creating board context once per board, then batching matchup evaluations. We apply the same pattern here.

---

## Experiment 19: Multi-threaded Exact Equity

**[Exact Equity]** Expected: 40-80× faster | Complexity: Medium | Status: Proposed

**Approach**: Parallelize board enumeration across CPU cores. Each thread processes a subset of boards independently. Board evaluations are embarrassingly parallel with no shared mutable state.

**Motivation**: After optimizing single-threaded performance with Experiment 18 (8-12× faster), the next multiplier is parallelism. The M1 CPU has 8 cores. Exact equity is embarrassingly parallel - each board evaluation is independent. The existing `equity.threaded()` function (equity.zig:841-911) already proves this pattern works for Monte Carlo.

**Implementation**:
```zig
const ExactThreadContext = struct {
    hero_hole_cards: Hand,
    villain_hole_cards: Hand,
    board_hand: Hand,
    board_completions: []const Hand,
    result: *EquityResult,
    wait_group: *std.Thread.WaitGroup,
};

fn exactWorkerThread(ctx: *ExactThreadContext) void {
    defer ctx.wait_group.finish();

    var wins: u32 = 0;
    var ties: u32 = 0;

    for (ctx.board_completions) |board_completion| {
        const complete_board = ctx.board_hand | board_completion;
        const board_ctx = evaluator.initBoardContext(complete_board);
        const result = evaluator.evaluateShowdownWithContext(&board_ctx, ctx.hero_hole_cards, ctx.villain_hole_cards);

        if (result > 0) {
            wins += 1;
        } else if (result == 0) {
            ties += 1;
        }
    }

    ctx.result.wins = wins;
    ctx.result.ties = ties;
    ctx.result.total_simulations = @intCast(ctx.board_completions.len);
}

pub fn exactThreaded(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand, allocator: std.mem.Allocator) !EquityResult {
    // ... existing validation and board enumeration ...

    const thread_count = @min(try std.Thread.getCpuCount(), 16);
    const boards_per_thread = board_completions.len / thread_count;

    var thread_results = try allocator.alloc(EquityResult, thread_count);
    defer allocator.free(thread_results);

    var contexts = try allocator.alloc(ExactThreadContext, thread_count);
    defer allocator.free(contexts);

    var wait_group = std.Thread.WaitGroup{};

    // Spawn worker threads
    for (0..thread_count) |thread_id| {
        const start_idx = thread_id * boards_per_thread;
        const end_idx = if (thread_id == thread_count - 1)
            board_completions.len
        else
            (thread_id + 1) * boards_per_thread;

        contexts[thread_id] = ExactThreadContext{
            .hero_hole_cards = hero_hole_cards,
            .villain_hole_cards = villain_hole_cards,
            .board_hand = board_hand,
            .board_completions = board_completions[start_idx..end_idx],
            .result = &thread_results[thread_id],
            .wait_group = &wait_group,
        };

        wait_group.start();
        _ = try std.Thread.spawn(.{}, exactWorkerThread, .{&contexts[thread_id]});
    }

    wait_group.wait();

    // Aggregate results
    var total_wins: u32 = 0;
    var total_ties: u32 = 0;
    var total_sims: u32 = 0;

    for (thread_results) |result| {
        total_wins += result.wins;
        total_ties += result.ties;
        total_sims += result.total_simulations;
    }

    return EquityResult{ .wins = total_wins, .ties = total_ties, .total_simulations = total_sims };
}
```

**Why it should work**:
- Board evaluations are embarrassingly parallel (no shared mutable state)
- Already proven pattern in `equity.threaded()` (equity.zig:841-911)
- M1 has 8 cores (4 performance + 4 efficiency) = potential 6-8× speedup
- Cache-line padding pattern already validated (equity.zig:779-801)
- Combines with Experiment 18 for multiplicative gains

**Expected benchmark** (AA vs KK preflop, building on Exp 18):
- **Baseline (after Exp 18)**: 0.7-1.1 seconds
- **With 8-core parallelism**: 110-220ms (6-8× additional speedup)
- **Total improvement**: 40-80× faster than original baseline

**Recommended approach**: Linear progression through Experiments 16 → 18 → 19
1. **Exp 16** (board context) validates API compatibility with minimal complexity
2. **Exp 18** (hybrid) achieves 8-12× baseline speedup for single-threaded workloads
3. **Exp 19** (threading) adds final 6-8× multiplier for production performance

**Expected final performance**: AA vs KK exact equity in **110-220ms** (vs current 8.67s), making exact equity **faster than 10,000-iteration Monte Carlo** while maintaining perfect accuracy.

---

## Experiment 20: SIMD Micro-Batch for Showdown Context

**[Context Path]** **✅ Success** | +2.34× faster (37.01→15.4 ns/eval) | Complexity: Low

**Motivation**: Profiling of real-world usage patterns revealed that `evaluateShowdownWithContext` is a critical bottleneck in production workloads. Current implementation performs two serial hand evaluations (hero and villain), missing SIMD and memory-level parallelism benefits. Profile data shows 96.5% of time spent in the two `evaluateHoleWithContextImpl` calls.

**Current Performance** (from profiling):
- `evaluateShowdownWithContext`: 37.01 ns/eval
- Batched showdown (Exp 10): 13.4 ns/eval
- **Gap**: 2.76× slower despite same underlying evaluation work

**Why it's slow**:
1. **Serial execution**: Two separate CHD lookups with no memory-level parallelism
2. **Array copies**: Each `evaluateHoleWithContextImpl` copies `suit_masks`, `suit_counts`, `rank_counts`
3. **Redundant RPC computation**: 13-iteration multiply-add loop runs twice
4. **No SIMD**: Scalar path misses vectorization benefits of batch path

**Approach**: Reuse existing `evaluateBatch` infrastructure with batch-2 SIMD path:

```zig
pub fn evaluateShowdownWithContext(ctx: *const BoardContext, hero_hole: card.Hand, villain_hole: card.Hand) i8 {
    std.debug.assert((hero_hole & villain_hole) == 0);
    std.debug.assert((hero_hole & ctx.board) == 0);
    std.debug.assert((villain_hole & ctx.board) == 0);

    // Pack both hands into batch-2 vector for SIMD evaluation
    const hands = @Vector(2, u64){
        ctx.board | hero_hole,
        ctx.board | villain_hole,
    };

    const ranks = evaluateBatch(2, hands);
    return if (ranks[0] < ranks[1]) @as(i8, 1)
           else if (ranks[0] > ranks[1]) @as(i8, -1)
           else @as(i8, 0);
}
```

**Implementation note**: Initial attempt with batch-4 and zero-filled unused lanes (`0, 0`) **failed** - resulted in 47 ns (slower than 37 ns baseline). The zeros cause issues in RPC computation and waste SIMD work. Batch-2 is the correct approach.

**Why it succeeded**:
- **Zero new code**: Reuses existing `evaluateBatch(2, ...)` implementation (Exp 6)
- **SIMD benefits**: Vectorized RPC and flush detection process both hands in parallel
- **Memory-level parallelism**: Overlaps CHD memory latency across lanes
- **Eliminates overhead**: No array copies, no redundant RPC loops
- **Optimal batch size**: Batch-2 perfectly fits the two-hand showdown use case

**Benchmark plan**:
```bash
# Build profiling tool
zig build -Doptimize=ReleaseFast

# Baseline measurement (50M showdowns)
./zig-out/bin/profile_context showdown 50000000

# After optimization
./zig-out/bin/profile_context showdown 50000000

# Full profile comparison
task profile:context:showdown
uniprof analyze /tmp/context_showdown_profile/profile.json
```

**Benchmark Results** (50M iterations, ReleaseFast, Apple M1):
- **Baseline (scalar)**: 37.01 ns/eval (27.0M evals/sec)
- **With batch-2 SIMD**: 15.4 ns/eval (64.9M evals/sec)
- **Improvement**: **2.34× faster** (21.6 ns reduction)
- **Consistency**: 3 runs: 15.10, 15.19, 15.41 ns (CV: 1.0%)
- **Tests**: All 121 tests passed

**Profile Analysis** (uniprof, 50M iterations):
- `detectFlushSimd`: 5.0% samples (SIMD flush detection)
- `computeRpcSimd`: 3.2% samples (vectorized RPC computation)
- `mphf.lookup`: 3.2% samples (CHD table lookups)
- Balanced distribution - no single bottleneck dominates

**Why it succeeded**:
1. **Memory-level parallelism**: Two CHD lookups can overlap L2 latency
2. **SIMD vectorization**: RPC computation vectorized across 2 hands
3. **Eliminates scalar overhead**: No array copies or redundant 13-iteration loops
4. **Compiler optimization**: Batch-2 inlines efficiently without register pressure

**Comparison to batched showdown** (Exp 10):
- Batched showdown (32-pair chunks): 13.4 ns/eval
- This optimization (single pair): 15.4 ns/eval
- **Gap**: Only 2 ns (15% overhead for single-pair vs bulk processing)
- Excellent result - near-optimal for the use case

**Production impact**:
- Showdown-heavy workloads: **2.34× throughput increase**
- Monte Carlo equity: Faster per-sample evaluation
- Simulation pipelines: Improved latency characteristics

**Key insight**: **Batch size matters**. Batch-4 with zeros failed because RPC computation treats zero as a valid hand (zero bits set = zero rank counts). Batch-2 perfectly matches the problem structure without waste.

---

## Experiment 21: Cached RPC Base in BoardContext

**[Context Path]** **Proposed** | Expected +1.3-1.6× faster (15→10-12 ns/eval after Exp 20) | Complexity: Medium

**Motivation**: Profiling revealed `computeBoardRankCounts` consumes 52% of `initBoardContext` time. Each `evaluateHoleWithContextImpl` then recomputes the full 13-iteration RPC loop. By caching the board-only RPC base and flush candidate mask, we can convert O(13) operations to O(1).

**Current bottleneck** (from profiling):
- `initBoardContext`: 10.38 ns/call
  - `computeBoardRankCounts`: ~52% of time (13×4=52 bit operations)
  - Used in RPC computation: 13-iteration multiply-add loop

**Approach**: Extend `BoardContext` with pre-computed data:

```zig
pub const BoardContext = struct {
    board: card.Hand,
    suit_masks: [4]u16,
    suit_counts: [4]u8,
    rank_counts: [13]u8,
    rpc_base: u32,              // NEW: pre-computed board-only RPC
    suit_flush_mask_ge3: u4,    // NEW: suits with ≥3 cards (flush candidates)
};
```

**Computation in `initBoardContext`**:
```zig
pub fn initBoardContext(board: card.Hand) BoardContext {
    const suit_masks = initSuitMasks(board);
    var suit_counts: [4]u8 = undefined;
    inline for (0..4) |suit| {
        suit_counts[suit] = @intCast(@popCount(suit_masks[suit]));
    }
    const rank_counts = computeBoardRankCounts(suit_masks);

    // NEW: Compute RPC base once (avoid 13-iteration loop per hole)
    var rpc_base: u32 = 0;
    for (rank_counts) |count| {
        rpc_base = rpc_base * 5 + count;
    }

    // NEW: Build flush candidate mask (only suits that can reach 5 cards)
    var flush_mask: u4 = 0;
    inline for (0..4) |suit| {
        if (suit_counts[suit] >= 3) {  // Can reach 5 with 2 hole cards
            flush_mask |= @as(u4, 1) << @intCast(suit);
        }
    }

    return .{
        .board = board,
        .suit_masks = suit_masks,
        .suit_counts = suit_counts,
        .rank_counts = rank_counts,
        .rpc_base = rpc_base,
        .suit_flush_mask_ge3 = flush_mask,
    };
}
```

**Optimized `evaluateHoleWithContextImpl`**:

```zig
fn evaluateHoleWithContextImpl(ctx: *const BoardContext, hole: card.Hand) HandRank {
    std.debug.assert((hole & ctx.board) == 0);

    // Extract hole cards (at most 2 cards)
    const bit1: u6 = @intCast(@ctz(hole));
    const bit2: u6 = @intCast(@ctz(hole & (hole - 1)));

    const suit1: usize = bit1 / 13;
    const rank1: usize = bit1 % 13;
    const suit2: usize = bit2 / 13;
    const rank2: usize = bit2 % 13;

    // Gated flush check: only test suits that can reach 5 cards
    const suit1_bit = @as(u4, 1) << @intCast(suit1);
    const suit2_bit = @as(u4, 1) << @intCast(suit2);

    if ((ctx.suit_flush_mask_ge3 & suit1_bit) != 0) {
        const count = ctx.suit_counts[suit1] + (if (suit1 == suit2) 2 else 1);
        if (count >= 5) {
            var mask = ctx.suit_masks[suit1];
            mask |= @as(u16, 1) << @intCast(rank1);
            if (suit1 == suit2) mask |= @as(u16, 1) << @intCast(rank2);
            const pattern = getTop5Ranks(mask);
            return tables.flushLookup(pattern);
        }
    }
    if (suit1 != suit2 and (ctx.suit_flush_mask_ge3 & suit2_bit) != 0) {
        if (ctx.suit_counts[suit2] + 1 >= 5) {
            const mask = ctx.suit_masks[suit2] | (@as(u16, 1) << @intCast(rank2));
            const pattern = getTop5Ranks(mask);
            return tables.flushLookup(pattern);
        }
    }

    // Incremental RPC: O(1) vs O(13)
    // Precompute pow5 table: [5^12, 5^11, ..., 5^0]
    const pow5 = [13]u32{
        244140625, 48828125, 9765625, 1953125, 390625,
        78125, 15625, 3125, 625, 125, 25, 5, 1,
    };

    const rpc = if (rank1 == rank2)
        ctx.rpc_base + 2 * pow5[rank1]
    else
        ctx.rpc_base + pow5[rank1] + pow5[rank2];

    return tables.lookup(rpc);
}
```

**Why it should succeed**:
- **Eliminates 13-iteration loop**: RPC computed incrementally in O(1)
- **Reduces flush checks**: Only test suits with ≥3 cards (skip 0-2 card suits)
- **Amortizes cost**: `initBoardContext` slightly slower, but saves work on every hole evaluation
- **Memory-friendly**: Adds 5 bytes to BoardContext (u32 + u4 + padding)

**Challenges**:
- **Division/modulo by 13**: May need bit_to_suit[52] and bit_to_rank[52] LUTs for optimal performance
- **Register pressure**: More variables in hot path (profile to confirm)
- **Code complexity**: More branches and conditionals

**Benchmark plan**:
```bash
# After Experiment 20
./zig-out/bin/profile_context showdown 50000000  # Baseline with SIMD
./zig-out/bin/profile_context initBoard 50000000 # Measure initBoardContext overhead

# After Experiment 21
./zig-out/bin/profile_context showdown 50000000  # With cached RPC
./zig-out/bin/profile_context initBoard 50000000 # Verify overhead acceptable
```

**Expected results** (combined with Exp 20):
- **After Exp 20**: 15-18 ns/eval
- **After Exp 21**: 10-12 ns/eval (additional 1.3-1.6× speedup)
- **initBoardContext**: 10.38→12-14 ns (acceptable for amortization)

**Validation**:
- Existing tests should pass unchanged (same BoardContext API surface)
- Add specific test for incremental RPC correctness
- Verify flush gating doesn't miss edge cases (7-card flush, wheel straight)

**Risk assessment**: **Medium**
- Complexity increase in hot path
- Potential for off-by-one errors in incremental RPC
- Division/modulo may still bottleneck (profile-guided decision)

---

## Experiment 22: Bit-to-Suit/Rank Lookup Tables

**[Context Path]** **Proposed** | Expected +1.2× faster (optional polish after Exp 21) | Complexity: Low

**Motivation**: If profiling Experiment 21 reveals division/modulo by 13 as a bottleneck, replace with direct lookup tables.

**Approach**: Replace arithmetic with table lookup:

```zig
// Compile-time lookup tables (104 bytes total)
const bit_to_suit = [52]u8{
    0,0,0,0,0,0,0,0,0,0,0,0,0,  // bits 0-12: clubs
    1,1,1,1,1,1,1,1,1,1,1,1,1,  // bits 13-25: diamonds
    2,2,2,2,2,2,2,2,2,2,2,2,2,  // bits 26-38: hearts
    3,3,3,3,3,3,3,3,3,3,3,3,3,  // bits 39-51: spades
};

const bit_to_rank = [52]u8{
    0,1,2,3,4,5,6,7,8,9,10,11,12,  // clubs
    0,1,2,3,4,5,6,7,8,9,10,11,12,  // diamonds
    0,1,2,3,4,5,6,7,8,9,10,11,12,  // hearts
    0,1,2,3,4,5,6,7,8,9,10,11,12,  // spades
};

// Replace division/modulo:
const suit1: usize = bit_to_suit[bit1];
const rank1: usize = bit_to_rank[bit1];
```

**When to implement**: Only if profiling shows division as bottleneck (>5% samples)

**Expected impact**: 1.1-1.2× if division is bottleneck, negligible if compiler already optimizes

---

## Implementation Roadmap (Context Path Optimization)

**Phase 1: Quick Win (Exp 20)** - Target: <1 day
1. Implement SIMD micro-batch for showdown
2. Benchmark and profile (expect 2-2.5× speedup)
3. Validate with existing tests

**Phase 2: Deeper Optimization (Exp 21)** - Target: 2-3 days
1. Extend BoardContext with rpc_base and flush_mask
2. Refactor evaluateHoleWithContextImpl for incremental RPC
3. Profile to verify gains and identify remaining bottlenecks
4. Validate correctness on edge cases

**Phase 3: Optional Polish (Exp 22)** - Target: <1 day
1. Profile Exp 21 to check division/modulo overhead
2. Implement bit_to_suit/rank LUTs if beneficial
3. Final benchmark comparison

**Expected cumulative impact**:
- **Current**: 37.01 ns/eval
- **After Exp 20**: 15-18 ns/eval (2.0-2.5× faster)
- **After Exp 21**: 10-12 ns/eval (3.1-3.7× faster total)
- **After Exp 22**: 9-11 ns/eval (3.4-4.1× faster total)

**Production benefits**:
- Faster Monte Carlo equity estimation
- Improved throughput for simulation-heavy workloads
- Better latency characteristics for real-time applications
