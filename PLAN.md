# Performance Optimization Plan

## Current Status
- **Performance**: 11.95 ns/hand (83.7M hands/s) on Apple M1
- **Goal**: Sub-10 ns/hand (100M+ hands/s) - need 17% improvement
- **Architecture**: CHD perfect hash + slow evaluator fallback for flushes

## Profiling Results (Validated Bottleneck)

High-frequency profiling with 3,886 samples confirms the bottleneck:

- **98.2%** of runtime in `evaluator.evaluate_hand` 
- **~95%** of samples hit **line 69**: `const rpc = compute_rpc_from_hand(hand);`
- **<2%** in hash/CHD lookup (`chd_lookup_scalar`)
- **Minor time** in flush detection and fallback

**Conclusion**: The 52-iteration rank counting nested loop dominates performance.

## Failed Optimization History

All previous attempts made performance **worse** (3.2% to 9.4% slower):

1. **Packed ARM64 tables** (+5.9% slower) - Memory not the bottleneck
2. **Memory prefetching** (+9.4% slower) - Cache-resident workload  
3. **Simplified hash function** (+3.2% slower) - Hash already optimized
4. **RPC bit manipulation** (+7.4% slower) - Compiler beats manual optimization

**Key Learning**: Complexity tax is real. Simple code + compiler optimization wins.

## o3-pro Analysis & Recommendations

### Root Cause Identification
- **Bottleneck**: 52-iteration rank counting loop in `compute_rpc_from_hand`
- **Not bottlenecks**: Memory access, hash computation, CHD lookup
- **Current "batch" processing**: Fake - just scalar calls in a loop

### Proposed Solution: LUT-Based Rank Counting

Replace the 52-iteration nested loop with O(1) table lookups:

#### **Phase 1: Scalar LUT Implementation**
```zig
// Pre-computed lookup tables (1.5KB total)
const rank_delta_low = [128]u32{ /* precomputed base-5 deltas for ranks 0-6 */ };
const rank_delta_high = [64]u32{ /* precomputed base-5 deltas for ranks 7-12 */ };

fn compute_rpc_lut(hand: u64) u32 {
    var rpc: u32 = 0;
    inline for (0..4) |suit| {
        const suit_mask = extract_suit(hand, suit);
        rpc += rank_delta_low[suit_mask & 0x7F];   // Low 7 bits
        rpc += rank_delta_high[suit_mask >> 7];    // High 6 bits  
    }
    return rpc;
}
```

**Benefits**:
- **No loops**: 4 suits × 2 LUT reads = 8 table accesses total
- **Cache-friendly**: 1.5KB tables fit in L1 cache
- **Vectorizable**: Four independent table lookups per suit
- **Same encoding**: Preserves existing CHD tables

#### **Phase 2: SIMD Implementation** (if scalar succeeds)
```zig
struct Hands4 {
    clubs: @Vector(4, u16),
    diamonds: @Vector(4, u16), 
    hearts: @Vector(4, u16),
    spades: @Vector(4, u16),
}

fn compute_rpc4_simd(hands4: Hands4) @Vector(4, u32) {
    // 4-lane parallel LUT lookups with NEON
    const rpc_vec = 
        rank_low_lut[hands4.clubs & 0x7F] +
        rank_high_lut[hands4.clubs >> 7] +
        rank_low_lut[hands4.diamonds & 0x7F] + 
        // ... etc for all suits
    return rpc_vec;
}
```

**Target Performance**: 4-lane SIMD could achieve 3.5-4.0 ns/hand

## Claude's Conservative Recommendation

Given the **consistent pattern of optimization failures**, recommend incremental approach:

### **Step 1: Profile-First Validation** ✅ COMPLETED
- Confirmed 52-iteration loop is the bottleneck
- Validated hash/lookup is NOT the bottleneck

### **Step 2: Minimal Scalar LUT Test**
1. Implement **scalar LUT version only** as drop-in replacement
2. Generate lookup tables in `build_tables.zig`
3. Benchmark against current implementation
4. **If slower, abandon immediately** (following EXPERIMENTS.md pattern)

### **Step 3: SIMD Only If Scalar Wins**
- Only attempt vectorization if scalar LUT shows improvement
- Start with simple data layout changes
- Avoid complex SoA conversions initially

## Implementation Plan

### **Phase 1: Scalar LUT (Low Risk)**

#### 1.1: Extend `build_tables.zig`
```zig
// Generate rank delta lookup tables
fn generateRankDeltaTables() void {
    var rank_delta_low: [128]u32 = undefined;
    var rank_delta_high: [64]u32 = undefined;
    
    for (0..128) |mask| {
        var delta: u32 = 0;
        for (0..7) |rank| {
            if (mask & (@as(u32, 1) << @intCast(rank)) != 0) {
                delta += std.math.pow(u32, 5, @intCast(rank));
            }
        }
        rank_delta_low[mask] = delta;
    }
    
    // Similar for rank_delta_high (ranks 7-12)
    // Write to tables.zig
}
```

#### 1.2: Implement `compute_rpc_lut`
Replace the nested loop with table lookups in `evaluator.zig`

#### 1.3: Benchmark & Validate
- Must beat 11.95 ns/hand baseline
- Must pass all correctness tests
- **If fails, revert and document in EXPERIMENTS.md**

### **Phase 2: SIMD (Higher Risk)**
Only proceed if Phase 1 succeeds:
1. Implement `Hands4` structure-of-arrays
2. Add SIMD LUT lookups 
3. Benchmark 4-hand batches

## Risk Assessment

### **Low Risk (Phase 1)**
- Scalar LUT is mathematically equivalent
- Small code change
- Easy to revert
- Tables only 1.5KB

### **Medium Risk (Phase 2)**  
- Complex data layout changes
- API breaking changes
- Vectorization assumptions may fail
- History shows manual optimization often loses

## Success Criteria

- **Phase 1**: Beat 11.95 ns/hand with scalar LUT
- **Phase 2**: Achieve <10 ns/hand with SIMD
- **Overall**: Maintain 100% correctness

## Fallback Strategy

If LUT approach fails:
- Document in EXPERIMENTS.md 
- Consider algorithmic alternatives (different evaluation method)
- Accept current 11.95 ns/hand as "good enough"
- Focus optimization efforts elsewhere

---

**Next Step**: Implement Phase 1 scalar LUT and benchmark against baseline.