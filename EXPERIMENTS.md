# Performance Optimization Experiments

**Baseline**: 11.95 ns/hand (83.7M hands/s) - Simple direct u16 table lookup

## Experiment 1: Packed ARM64 Tables
**Performance Impact**: +5.9% slower (12.65→11.95 ns/hand)  
**Complexity**: High  
**Status**: ❌ Failed  

**Approach**: Pack two 13-bit hand ranks into each u32 table entry to halve memory footprint on ARM64. Added architecture detection in build_tables.zig and dual table formats with bit manipulation for extraction.

**Implementation**: 
- Modified CHD value table from `[CHD_TABLE_SIZE]u16` to `[CHD_TABLE_SIZE/2]u32`
- Added `unpack_rank_arm64()` function with bit shifts and masks
- Conditional table generation based on target architecture

**Why it failed**: Complexity overhead outweighed theoretical memory benefits. The 267KB working set already fits comfortably in L2 cache, so halving it provided no cache benefit. The bit manipulation added CPU cycles without reducing memory pressure.

## Experiment 2: Memory Prefetching  
**Performance Impact**: +9.4% slower (13.07→11.95 ns/hand)  
**Complexity**: Medium  
**Status**: ❌ Failed  

**Approach**: Add `@prefetch` instructions in batch evaluation loops to hint upcoming memory accesses to the CPU cache system.

**Implementation**:
```zig
@prefetch(&tables.chd_value_table[final_index], .{});
```

**Why it failed**: The 267KB table working set is cache-resident, not memory-bound. Prefetch hints added overhead without improving cache hit rates. The evaluator is compute-bound, not memory-bound.

## Experiment 3: Simplified Hash Function
**Performance Impact**: +3.2% slower (12.33→11.95 ns/hand)  
**Complexity**: Low  
**Status**: ❌ Failed  

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

## Experiment 4: RPC Bit Manipulation
**Performance Impact**: +7.4% slower (12.83→11.95 ns/hand)  
**Complexity**: Medium  
**Status**: ❌ Failed  

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

## Experiment 5: LUT-Based RPC Computation
**Performance Impact**: +10.2% slower (14.10→12.79 ns/hand)  
**Complexity**: High  
**Status**: ❌ Failed  

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

## Experiment 6: SIMD Batching (4-Hand Parallel)
**Performance Impact**: +89% faster (7.22→3.82 ns/hand)  
**Complexity**: High  
**Status**: ✅ SUCCESS  

**Approach**: Process 4 poker hands simultaneously using true SIMD parallelism with ARM64 NEON via Zig @Vector types, rather than optimizing single-hand computation.

**Implementation**:
```zig
// Structure-of-arrays for SIMD processing
const Hands4 = struct {
    clubs: @Vector(4, u16),
    diamonds: @Vector(4, u16), 
    hearts: @Vector(4, u16),
    spades: @Vector(4, u16),
};

// Vectorized rank counting
fn compute_rpc_simd4(hands4: Hands4) @Vector(4, u32) {
    var rpc_vec: @Vector(4, u32) = @splat(0);
    
    inline for (0..13) |rank| {
        const rank_bit: @Vector(4, u16) = @splat(@as(u16, 1) << @intCast(rank));
        const zero_vec: @Vector(4, u16) = @splat(0);
        
        // Count rank occurrences across all suits (vectorized)
        const clubs_has = @select(u8, (hands4.clubs & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const diamonds_has = @select(u8, (hands4.diamonds & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const hearts_has = @select(u8, (hands4.hearts & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        const spades_has = @select(u8, (hands4.spades & rank_bit) != zero_vec, one_vec, zero_u8_vec);
        
        const rank_count_vec = clubs_has + diamonds_has + hearts_has + spades_has;
        
        // Vectorized base-5 encoding
        const five_vec: @Vector(4, u32) = @splat(5);
        rpc_vec = rpc_vec * five_vec + @as(@Vector(4, u32), rank_count_vec);
    }
    
    return rpc_vec;
}
```

**Benchmark Results**: 4M hands (1M batches of 4)
- **Scalar**: 7.22 ns/hand (138.4 M hands/sec)
- **SIMD**: 3.82 ns/hand (262.0 M hands/sec)  
- **Speedup**: 1.89x
- **Checksums**: Identical (correctness verified)

**Why it succeeded**: Instead of fighting compiler optimizations with single-hand complexity, leveraged algorithmic parallelism. ARM64 NEON SIMD units can genuinely process 4 values simultaneously, providing true throughput multiplication. The structure-of-arrays layout enables efficient vectorization of the rank counting bottleneck.

**Key insight**: All previous experiments failed because they added complexity to already-optimized single-hand code. This experiment succeeded by changing the **algorithm** from "optimize 1 hand" to "process 4 hands in parallel" - working with SIMD strengths rather than against compiler optimizations.

## Key Learnings

1. **Memory optimizations fail**: Working set (267KB) is cache-resident, not memory-bound
2. **Complexity tax is real**: Small theoretical gains eaten by implementation overhead  
3. **Compiler knows best**: Auto-vectorization + simple code often beats manual optimization
4. **Measure everything**: "Obvious" optimizations frequently backfire
5. **Profile-guided is better**: All experiments were theory-driven; real profiling might reveal different bottlenecks
6. **Algorithmic parallelism wins**: True SIMD batching succeeds where single-hand optimizations fail

---

## Test Environment

**Hardware**: MacBook Air (M1, 2020)  
**CPU**: Apple M1 (8-core, 4 performance + 4 efficiency)  
**Memory**: 16 GB unified memory  
**OS**: macOS 14.2.1 (23C71)  
**Compiler**: Zig 0.14.0  
**Build flags**: `-Doptimize=ReleaseFast -Dcpu=native`  

*All experiments measured with 5 benchmark runs of 100K batches (400K hands) each*