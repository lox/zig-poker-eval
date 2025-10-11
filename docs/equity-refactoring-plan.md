# Equity Module Refactoring Plan

**Date:** 2025-10-11
**Current LOC:** 1247 lines
**Target LOC:** ~1050 lines (16% reduction)
**Goal:** Consolidate duplicate code while maintaining 100% performance

## Performance Baseline (Apple M1, ReleaseFast)

### Hand Evaluation
```
Batch performance:   3.83 ns/hand
Hands per second:    261.2M hands/sec
```

### Showdown Evaluation
```
Context path:        30.53 ns/eval
Batched path:        7.88 ns/eval
Speedup:             3.87x
```

### Equity Calculations
```
Scenario           | Simulations | Time (ms) | Sims/sec (millions)
-------------------|-------------|-----------|--------------------
Preflop (no board) |       10000 |      0.71 |              14.03
Flop (3 cards)     |        5000 |      0.12 |              40.98
Turn (4 cards)     |        2000 |      0.03 |              64.52
River (5 cards)    |        1000 |      0.01 |             142.86
```

### Multi-way Equity
```
3 players          |        5000 |      0.53 |               9.52
4 players          |        5000 |      0.53 |               9.47
6 players          |        5000 |      0.58 |               8.56
9 players          |        5000 |      1.06 |               4.71
```

## Refactoring Phases

### Phase 1: Merge Result Types (Low Risk)

**Target:** Lines 14-146
**Savings:** ~70 lines
**Risk Level:** 🟢 Low

#### Current State
- `EquityResult` (basic, 35 lines)
- `DetailedEquityResult` (with categories + CI, 43 lines)
- `DetailedExactResult` (with categories, no CI, 26 lines)

#### Proposed Change
Merge into single unified type:

```zig
pub const EquityResult = struct {
    wins: u32,
    ties: u32,
    total_simulations: u32,
    hand1_categories: ?HandCategories = null,
    hand2_categories: ?HandCategories = null,

    pub fn equity(self: EquityResult) f64 { /* ... */ }
    pub fn winRate(self: EquityResult) f64 { /* ... */ }
    pub fn tieRate(self: EquityResult) f64 { /* ... */ }
    pub fn lossRate(self: EquityResult) f64 { /* ... */ }

    /// Returns confidence interval for Monte Carlo results
    /// Returns null for exact calculations or non-detailed results
    pub fn confidenceInterval(self: EquityResult) ?struct { lower: f64, upper: f64 } {
        if (self.hand1_categories == null) return null;
        // existing CI calculation
    }
};
```

#### Migration Strategy
1. Add unified `EquityResult` type
2. Add type aliases for backward compatibility:
   ```zig
   pub const DetailedEquityResult = EquityResult;
   pub const DetailedExactResult = EquityResult;
   ```
3. Update all function signatures to return unified type
4. Remove old types in next major version

#### Performance Impact
- Memory: Minimal (~16 bytes per result for null pointers when categories not tracked)
- CPU: Zero (nullable types are compile-time overhead only)

#### Validation
```bash
zig build test
./zig-out/bin/poker-eval bench --equity
```

### Phase 2: Consolidate Sampling Functions (Low Risk)

**Target:** Lines 719-771
**Savings:** ~30 lines
**Risk Level:** 🟢 Low

#### Current State
- `sampleRemainingCardsHeadToHead` (lines 719-737)
- `sampleRemainingCardsForEquity` (lines 740-766)
- `sampleRemainingCardsForEquityDirect` (lines 769-771) - trivial wrapper

#### Proposed Change
Single unified function:

```zig
/// Sample remaining cards for equity calculations
/// @param hands Array of hole card bitmasks (can be empty for single-hand sampling)
/// @param board Current board bitmask
/// @param num_cards Number of cards to sample
fn sampleRemainingCards(hands: []const Hand, board: Hand, num_cards: u8, rng: Random) Hand {
    var used_bits = board;
    for (hands) |h| used_bits |= h;

    var sampled: u64 = 0;
    var cards_sampled: u8 = 0;
    while (cards_sampled < num_cards) {
        const card_idx = rng.uintLessThan(u8, 52);
        const card_bit = @as(u64, 1) << @intCast(card_idx);
        if ((used_bits & card_bit) == 0 and (sampled & card_bit) == 0) {
            sampled |= card_bit;
            cards_sampled += 1;
        }
    }
    return sampled;
}
```

#### Call Site Updates
- `monteCarlo` line 346: Pass `&[_]Hand{hero_hole_cards, villain_hole_cards}`
- `detailedMonteCarlo` line 346: Same
- `multiway` line 640: Already uses slice

#### Performance Impact
- Zero - same algorithm, just unified
- Compiler will inline the function

#### Validation
```bash
zig build test
./zig-out/bin/poker-eval bench --equity --verbose
```

### Phase 3: Remove Trivial Wrappers (Low Risk)

**Target:** Lines 488-490, 702-714
**Savings:** ~20 lines
**Risk Level:** 🟢 Low

#### Functions to Remove
1. `exactHeadToHead` (lines 488-490)
   - Just calls: `exact(hero_hole_cards, villain_hole_cards, &.{}, allocator)`
   - Users can call `exact` directly with empty board slice

2. `heroVsFieldMonteCarlo` (lines 702-714)
   - Just builds array and calls `multiway`
   - Users can easily do this themselves

#### Migration Strategy
1. Mark as deprecated with compiler warning:
   ```zig
   /// @deprecated Use exact() with empty board slice: exact(hero, villain, &.{}, allocator)
   pub fn exactHeadToHead(...) !EquityResult {
       return exact(hero_hole_cards, villain_hole_cards, &.{}, allocator);
   }
   ```

2. Update documentation with migration guide

3. Remove in next major version

#### Performance Impact
- None - these are just convenience wrappers

#### Validation
- Update any internal tests that use these functions
- Grep for usage: `rg "exactHeadToHead|heroVsFieldMonteCarlo"`

### Phase 4: Consolidate Calculation Functions (Medium Risk)

**Target:** Lines 210-376 (monteCarlo + detailedMonteCarlo), 381-481 (exact + exactDetailed)
**Savings:** ~80 lines
**Risk Level:** 🟡 Medium - Requires careful testing

#### Current State
Each calculation exists in two variants:
- Basic (no category tracking)
- Detailed (with category tracking)

The only difference is category tracking code.

#### Proposed Change
Use `comptime` parameters for zero-cost abstraction:

```zig
fn monteCarloImpl(
    comptime track_categories: bool,
    hero_hole_cards: Hand,
    villain_hole_cards: Hand,
    board: []const Hand,
    simulations: u32,
    rng: Random,
    allocator: Allocator,
) !EquityResult {
    // Validation (shared)
    if (card.countCards(hero_hole_cards) != 2) return error.InvalidHeroHoleCards;
    if (card.countCards(villain_hole_cards) != 2) return error.InvalidVillainHoleCards;
    if ((hero_hole_cards & villain_hole_cards) != 0) return error.ConflictingHoleCards;

    var board_hand: Hand = 0;
    for (board) |board_card| board_hand |= board_card;
    const cards_needed = 5 - @as(u8, @intCast(board.len));

    var wins: u32 = 0;
    var ties: u32 = 0;
    var hand1_categories = if (track_categories) HandCategories{} else null;
    var hand2_categories = if (track_categories) HandCategories{} else null;

    // Core logic (shared)
    const used_mask = hero_hole_cards | villain_hole_cards | board_hand;
    var available_cards: [52]u8 = undefined;
    var available_count: u8 = 0;
    for (0..52) |i| {
        const card_bit = @as(u64, 1) << @intCast(i);
        if ((card_bit & used_mask) == 0) {
            available_cards[available_count] = @intCast(i);
            available_count += 1;
        }
    }

    // SIMD batching (shared between both variants)
    const BATCH_SIZE = 16;
    const num_batches = simulations / BATCH_SIZE;
    const remainder = simulations % BATCH_SIZE;

    var batch_idx: u32 = 0;
    while (batch_idx < num_batches) : (batch_idx += 1) {
        var hero_batch: @Vector(BATCH_SIZE, u64) = undefined;
        var villain_batch: @Vector(BATCH_SIZE, u64) = undefined;

        inline for (0..BATCH_SIZE) |i| {
            var sampled: u64 = 0;
            var cards_sampled: u8 = 0;
            while (cards_sampled < cards_needed) {
                const idx = rng.uintLessThan(u8, available_count);
                const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
                if ((sampled & card_bit) == 0) {
                    sampled |= card_bit;
                    cards_sampled += 1;
                }
            }
            const complete_board = board_hand | sampled;
            hero_batch[i] = hero_hole_cards | complete_board;
            villain_batch[i] = villain_hole_cards | complete_board;
        }

        const hero_ranks = evaluator.evaluateBatch(BATCH_SIZE, hero_batch);
        const villain_ranks = evaluator.evaluateBatch(BATCH_SIZE, villain_batch);

        inline for (0..BATCH_SIZE) |i| {
            // Category tracking (comptime branching - zero runtime cost)
            if (track_categories) {
                hand1_categories.?.addHand(evaluator.getHandCategory(hero_ranks[i]));
                hand2_categories.?.addHand(evaluator.getHandCategory(villain_ranks[i]));
            }

            if (hero_ranks[i] < villain_ranks[i]) {
                wins += 1;
            } else if (hero_ranks[i] == villain_ranks[i]) {
                ties += 1;
            }
        }
    }

    // Handle remainder (shared)
    for (0..remainder) |_| {
        var sampled: u64 = 0;
        var cards_sampled: u8 = 0;
        while (cards_sampled < cards_needed) {
            const idx = rng.uintLessThan(u8, available_count);
            const card_bit = @as(u64, 1) << @intCast(available_cards[idx]);
            if ((sampled & card_bit) == 0) {
                sampled |= card_bit;
                cards_sampled += 1;
            }
        }
        const complete_board = board_hand | sampled;
        const hero_hand = hero_hole_cards | complete_board;
        const villain_hand = villain_hole_cards | complete_board;

        const hero_rank = evaluator.evaluateHand(hero_hand);
        const villain_rank = evaluator.evaluateHand(villain_hand);

        if (track_categories) {
            hand1_categories.?.addHand(evaluator.getHandCategory(hero_rank));
            hand2_categories.?.addHand(evaluator.getHandCategory(villain_rank));
        }

        if (hero_rank < villain_rank) {
            wins += 1;
        } else if (hero_rank == villain_rank) {
            ties += 1;
        }
    }

    return EquityResult{
        .wins = wins,
        .ties = ties,
        .total_simulations = simulations,
        .hand1_categories = hand1_categories,
        .hand2_categories = hand2_categories,
    };
}

/// Basic Monte Carlo (no category tracking)
pub fn monteCarlo(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand,
                  simulations: u32, rng: Random, allocator: Allocator) !EquityResult {
    _ = allocator; // Mark as unused
    return monteCarloImpl(false, hero_hole_cards, villain_hole_cards, board, simulations, rng, allocator);
}

/// Detailed Monte Carlo (with category tracking)
pub fn monteCarloDetailed(hero_hole_cards: Hand, villain_hole_cards: Hand, board: []const Hand,
                         simulations: u32, rng: Random, allocator: Allocator) !EquityResult {
    _ = allocator; // Mark as unused
    return monteCarloImpl(true, hero_hole_cards, villain_hole_cards, board, simulations, rng, allocator);
}
```

Apply the same pattern to `exact` and `exactDetailed`.

#### Performance Impact
- **Zero runtime cost** - `comptime` branches are eliminated at compile time
- Generates two separate optimized code paths
- SIMD optimizations preserved in both paths

#### Validation Strategy
**Critical:** This requires extensive validation

1. **Unit tests:**
   ```bash
   zig build test -Doptimize=ReleaseFast
   ```

2. **Benchmark comparison (must match baseline ±2%):**
   ```bash
   ./zig-out/bin/poker-eval bench --equity > after-refactor.txt
   diff baseline.txt after-refactor.txt
   ```

3. **Correctness validation:**
   ```bash
   # Run 100K random equity calculations, compare results
   ./zig-out/bin/poker-eval test-equity-correctness
   ```

4. **Profile to verify no deoptimization:**
   ```bash
   task profile:eval
   # Check that monteCarloImpl is properly inlined
   ```

### Phase 5: Documentation Updates (Low Risk)

**Target:** Update all relevant docs
**Risk Level:** 🟢 Low

#### Files to Update
1. `README.md` - Update API examples
2. `docs/api.md` - Document unified result type
3. `docs/CHANGELOG.md` - Add migration guide
4. Inline documentation - Update @deprecated tags

#### Example Migration Guide
```markdown
## Migrating to Unified EquityResult

### Before
```zig
const basic = try poker.monteCarlo(...);
const detailed = try poker.detailedMonteCarlo(...);
// basic is EquityResult, detailed is DetailedEquityResult
```

### After
```zig
const basic = try poker.monteCarlo(...);
const detailed = try poker.monteCarloDetailed(...);
// Both return EquityResult
// detailed.hand1_categories is non-null
// detailed.confidenceInterval() returns ?struct
```
```

## Implementation Order

1. ✅ **Capture performance baseline** (completed above)
2. 🔄 **Phase 1:** Merge result types (1-2 hours)
3. 🔄 **Phase 2:** Consolidate sampling (1 hour)
4. 🔄 **Phase 3:** Deprecate wrappers (30 minutes)
5. 🔄 **Phase 4:** Consolidate calculations (3-4 hours)
6. 🔄 **Phase 5:** Update documentation (1 hour)

**Total estimated time:** 7-9 hours

## Success Criteria

### Must Have
- ✅ All existing tests pass
- ✅ Performance within 2% of baseline (14.03M sims/sec preflop)
- ✅ No breaking API changes (deprecations only)
- ✅ Code reduction of at least 150 lines

### Should Have
- ✅ Improved code clarity and maintainability
- ✅ Reduced cognitive load (fewer types to understand)
- ✅ Better inline documentation

### Nice to Have
- Performance improvement from better inlining
- Reduced binary size from code deduplication

## Rollback Plan

Each phase is independent:
- Phase 1-3: Can be rolled back individually via git revert
- Phase 4: Keep both implementations until validation passes
- All phases: Feature flag for A/B testing if needed

## Performance Validation Checkpoints

After each phase:
```bash
# Run full benchmark suite
./zig-out/bin/poker-eval bench --equity
./zig-out/bin/poker-eval bench --showdown

# Validate against baseline
python scripts/compare-benchmarks.py baseline.json current.json

# If regression > 2%: investigate before proceeding
```

## Risk Mitigation

### High-Risk Changes (Phase 4)
- ✅ Keep original implementations commented out during testing
- ✅ Use comptime to ensure zero runtime branching
- ✅ Extensive benchmark coverage before committing
- ✅ Profile to verify inlining occurs

### Medium-Risk Changes (Phase 1)
- ✅ Add type aliases for backward compatibility
- ✅ Run full test suite after each change
- ✅ Document nullable field behavior

### Low-Risk Changes (Phase 2-3)
- ✅ Simple mechanical refactoring
- ✅ Compiler catches most issues
- ✅ Easy to review in code review

## Testing Strategy

### Automated Tests
```bash
# Run all tests with optimizations
zig build test -Doptimize=ReleaseFast

# Specific equity tests
zig build test -Doptimize=ReleaseFast --test-filter "equity"

# Benchmark validation
./zig-out/bin/poker-eval bench --validate --equity
```

### Manual Verification
```bash
# Known equity calculations (must match exactly)
./zig-out/bin/poker-eval equity "AhAs" "KdKc" --exact
# Should be: ~82.4%

./zig-out/bin/poker-eval equity "AhAs" "KdKc" --sims 1000000
# Should be: ~82.4% ±0.2%
```

## Notes

- **Comptime is key:** All branching for category tracking must be comptime
- **SIMD preserved:** The batch evaluation paths must remain unchanged
- **Backward compatible:** Existing code continues to work via type aliases
- **Performance critical:** Any regression > 2% requires investigation

## Future Optimization Opportunities

After consolidation is complete and stable:

1. **Threaded implementation consolidation**
   - Lines 865-1000 could potentially use same comptime pattern
   - Risk: Higher complexity, should be separate effort

2. **Multiway optimization**
   - Could benefit from same techniques
   - Keep separate for now to limit scope

3. **Alternative sampling algorithms**
   - Fisher-Yates implementation
   - Rejection sampling improvements
   - Separate performance experiment

## References

- Original analysis: gpt-5-pro thinkdeep session (continuation_id: 1bc94ce9-a0f6-497d-9efc-06a6ed7e13ae)
- Zig comptime documentation: https://ziglang.org/documentation/master/#comptime
- Current performance baseline: See top of this document
