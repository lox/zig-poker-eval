# Benchmarking Methodology

## Hardware and OS Setup

1. Pick one platform, document model, firmware version, and memory configuration.

2. Disable turbo boost and Hyper-Threading in macOS:
   - Use `pmset -g` to check power management settings
   - Set CPU frequency governor: `sudo pmset -a highperf 1`
   - Pin benchmark to specific CPU cores using thread affinity APIs

3. Verify fixed clock frequency:
   - Use `rdtsc` before/after 10s pause to confirm consistent TSC rate
   - Check thermal state with `pmset -g thermlog`

4. Minimize system interference:
   - Disable background processes: `sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.metadata.mds.plist`
   - Run benchmark with elevated priority: `sudo nice -20`

## Build Discipline

1. Clone evaluator code in a fresh directory.

2. Build in release mode:
   ```
   zig build bench -Doptimize=ReleaseFast -Dcpu=native
   ```

3. Record:
   - Zig version and build flags
   - macOS version and Xcode command line tools version
   - Binary SHA256 hash for repeatability

## Workload Generation

**Random** means each lookup is an independent 7-card set with no shared prefix; the generation of those keys must not appear in the timed region.

1. **Offline step**: Generate 2 × 10⁸ 64-bit masks with fixed seed xoshiro256**, store in flat array
   - 2e8 × 8B = 1.6GB; fits in RAM and forces occasional LLC evictions when streamed repeatedly

2. Shuffle the array once with Fisher-Yates; keep index order fixed across runs

3. **Cache-warm phase**: Touch first 1MB of lookup tables and first 64k hands so early misses aren't counted

## Timer and Counters

Use `rdtsc` to read TSC at batch boundaries; amortize call overhead by evaluating in blocks of 1,048,576 hands.

For each run capture:
- Elapsed TSC, convert to ns with calibrated clock rate
- Instructions, cache misses, branch mispredicts (using macOS performance counters)
- Memory allocation traffic (should be zero)
- Resident set size after memory locking

Store everything as CSV row.

## Run Plan

1. **Single-thread scalar path** (no SIMD) as baseline

2. **Preferred SIMD width** (AVX-512 or AVX2)

3. Repeat each of (1) and (2) five times; between repetitions:
   - Clear caches: `sudo purge`
   - Sleep 3 seconds

4. Run same harness with trivial "dummy evaluator" that returns popcount of mask
   - Measures framework/RNG/memory overhead
   - Subtract from real numbers to isolate algorithm time

5. **Optional**: Run sequential-access variant where mask array is sorted
   - Quantifies cache-hit headroom the algorithm could exploit

## Metrics and Analysis

**Primary metric**: Median nanoseconds per hand across five runs (batch time / 1,048,576), after overhead correction.

**Secondary metrics**:
- Cycles per hand (ns × nominal GHz)
- Instructions per hand (sanity-check against source; huge swings indicate inlining issues)
- L1/L2/L3 MPKI (validates cache-fit claims)
- Standard deviation among repetitions; target CV < 2%

Plot nanoseconds per hand vs. SIMD width, and MPKI vs. algorithm to show whether speed comes from fewer instructions or better memory locality.

## Correctness Guard

During warm-up pass:
- Compute 64-bit rolling XOR of all returned ranks
- Compare against golden value pre-computed with slow, trusted evaluator
- Abort run on first mismatch

## Reporting

Publish:
- CPU/firmware details, compiler, exact git hashes
- Complete build flags, performance counters used
- Full CSV and plotting script (gnuplot or Python)
- Raw array of random hands (or SHA256) so anyone can rerun identically

With this apparatus you get sub-3% measurement noise, isolate cache effects, and can defend claims such as "4.3ns ±0.05ns per random hand on M2 Pro, 3.5GHz".
