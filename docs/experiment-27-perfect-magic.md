# Experiment 27: Perfect Magic Search

**Status:** Running
**Started:** 2025-10-25
**Expected Duration:** 12-48 hours

## Goal

Find a single "perfect" magic constant `M` such that:

```zig
∀ rpc ∈ {49,205 unique RPCs}: (rpc * M) >> 47 → unique index ∈ [0, 131,071]
```

This would eliminate the CHD displacement array (`g_array`), reducing memory accesses from 3 to 1.

## Current Implementation (CHD)

```zig
pub inline fn lookup(key: u32, magic: u64, g_array: []const u8,
                     value_table: []const u16, table_size: u32) u16 {
    const h = mix64(@as(u64, key), magic);
    const bucket = @intCast(u32, h >> 51);              // 1st access
    const displacement = g_array[bucket];                // 2nd access (8KB, L1)
    const base_index = @intCast(u32, h & 0x1FFFF);
    const final_index = (base_index + displacement) & (table_size - 1);
    return value_table[final_index];                    // 3rd access (256KB, L2)
}
```

**Performance:** 2.8-3.8ns per lookup (3 dependent memory accesses)

## Target Implementation (Perfect Magic)

```zig
pub inline fn lookupPerfect(key: u32, magic: u64, value_table: []const u16) u16 {
    const h = @as(u64, key) *% magic;
    const index = @intCast(u32, h >> 47);
    return value_table[index];  // Single memory access!
}
```

**Expected Performance:** 2.3-3.0ns per lookup (1 memory access)
**Expected Speedup:** 15-30% faster

## Motivation

Inspired by [Steinar Gunderson's blog post](https://blog.sesse.net/blog/tech/2025-10-23-21-23_modern_perfect_hashing.html) on modern perfect hashing for strings, which achieved 2× speedup over gperf by:

1. Finding magic constants via brute force search
2. Using simple multiply + shift hash
3. Eliminating indirection layers

Chess magic bitboards prove this approach works for similar problem sizes (4K-8K patterns).

## Technical Details

### Problem Space

- **Keys:** 49,205 unique RPCs (rank pattern codes)
- **Table size:** 131,072 slots (2^17)
- **Load factor:** 37.5% (plenty of room)
- **Search space:** 2^64 possible magic constants

### Search Strategy

1. **Random candidate generation:** Generate odd numbers (better for multiplication)
2. **Killer heuristic:** Track most-frequent colliding pairs, test them first
3. **Parallel search:** Multiple threads search independently
4. **Early termination:** First thread to find a magic wins

### Expected Search Times

- **Optimistic:** 2^30 attempts (~10 seconds at 100M/sec)
- **Realistic:** 2^40 attempts (~3 hours at 100M/sec)
- **Pessimistic:** 2^50 attempts (~13 days at 100M/sec)

### Killer Heuristic

Borrowed from computer chess magic bitboard generation:

- Track which RPC pairs collide most frequently
- Test these pairs first on each new magic candidate
- Fast rejection saves ~99% of full collision checks
- Significant speedup over naive approach

## Running the Experiment

### Quick Start

```bash
# Default: 16 threads, 12-hour timeout
task exp27:find-magic

# Custom configuration
task exp27:find-magic THREADS=32 TIMEOUT=24

# Or directly via zig build
zig build find-magic -Doptimize=ReleaseFast -- --threads 32 --timeout 24
```

### Monitoring Progress

The tool reports progress every 10 seconds:

```
[Thread 0] 15.3 million attempts in 120.5s (127.1M/s)
[Thread 1] 14.8 million attempts in 120.5s (122.8M/s)
...
```

### Success Output

If a magic is found:

```
🎉 Thread 5 FOUND PERFECT MAGIC: 0xA5F9B3D1C8E74620

═══════════════════════════════════════════════════════════
  SUCCESS! 🎉
═══════════════════════════════════════════════════════════

Found perfect magic: 0xA5F9B3D1C8E74620

Search statistics:
  Total attempts:  2,847,193,582
  Elapsed time:    2.3 hours
  Search rate:     343.2 million/sec
```

### Timeout Output

If no magic is found:

```
═══════════════════════════════════════════════════════════
  TIMEOUT
═══════════════════════════════════════════════════════════

No perfect magic found in 12 hours

Search statistics:
  Total attempts:  5,183,947,291,837
  Elapsed time:    12.0 hours
  Search rate:     119.8 million/sec
  Search space:    ~2^42.2

Conclusion:
  - CHD was the right choice
  - Two-level hashing more reliable than perfect magic
```

## Next Steps If Successful

### 1. Verify Correctness

```bash
# Modify build_tables.zig to use perfect magic
# Rebuild tables and run full verification
task verify:all-hands
```

### 2. Benchmark Performance

```bash
# Compare lookupPerfect() vs current CHD lookup()
task bench
```

### 3. Decision Criteria

**Adopt if:**
- ✅ Found in <12 hours
- ✅ Speedup >20% in benchmarks
- ✅ All 133M hands verify correctly

**Keep CHD if:**
- ❌ Not found in 24 hours
- ❌ Speedup <10%
- ❌ Any correctness issues

## Risk Assessment

### Low Risk

- **Can always fall back to CHD:** Current implementation keeps working
- **Construction is offline:** Even 24-hour search is acceptable
- **Cache the magic:** Only search once, use forever

### Potential Issues

1. **No magic exists:** Unlikely at 37.5% load factor, but possible
2. **Search takes too long:** Set reasonable timeout (12-48 hours)
3. **Speedup less than expected:** Memory-bound may limit gains
4. **Platform-specific:** Magic might differ per architecture (unlikely)

## Theory vs Practice

### Why This Should Work

1. **Low load factor:** 37.5% gives lots of collision-free space
2. **Proven technique:** Chess magics work for similar problem sizes
3. **Simple math:** Multiply + shift is fast and well-studied
4. **One-time cost:** Construction time doesn't matter

### Why This Might Fail

1. **RPC distribution:** Our keys might have pathological distribution
2. **Table size constraint:** 2^17 might be too small (could try 2^18)
3. **Search space too large:** 2^64 is enormous, might need luck
4. **Memory-bound anyway:** L2 cache latency may dominate regardless

## Related Work

- **Chess magic bitboards:** [Fancy magic bitboards](https://www.chessprogramming.org/Magic_Bitboards)
- **Sesse's blog:** [Modern perfect hashing](https://blog.sesse.net/blog/tech/2025-10-23-21-23_modern_perfect_hashing.html)
- **Academic survey:** [Modern Minimal Perfect Hashing](https://arxiv.org/abs/2506.06536)

## Experiment Log

### 2025-10-25: Experiment Started

- Created `src/tools/find_perfect_magic.zig`
- Added build step: `zig build find-magic`
- Added task: `task exp27:find-magic`
- Configuration: 16 threads, 12-hour timeout
- Server: [Server details to be added]

### Results: TBD

Results will be documented here when search completes.

## Conclusion: TBD

This experiment will either:

1. **Find a perfect magic** → Document speedup and adopt if >20%
2. **Timeout** → Confirm CHD was the right algorithmic choice
3. **Provide insights** → Either way, we learn something valuable

The journey is the destination. Even if we don't find a magic, we'll have:
- Validated our CHD implementation choice
- Explored the limits of perfect hashing for this problem
- Created reusable magic-search infrastructure
- Added valuable data to the perfect hashing literature

---

**Update this document with results when the search completes!**
