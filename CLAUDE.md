# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Activate Hermit environment (required for Zig)
. bin/activate-hermit

# Build and run (debug mode)
zig build run

# Build and run optimized (production performance)
zig build run -Doptimize=ReleaseFast

# Run tests
zig build test

# Run comprehensive performance profiling
zig build profile
zig build profile -Doptimize=ReleaseFast  # For release mode profiling
```

## Architecture Overview

This is a high-performance 7-card Texas Hold'em poker hand evaluator achieving 48+ million evaluations per second. The core design uses bit manipulation and optimized flush detection for maximum performance:

- **Cards as bits**: Each card represented as a single bit in a u64 bitfield
- **Zero-cost abstractions**: Extensive use of `inline for` and `@popCount` for CPU-native operations
- **Branchless evaluation**: Optimized control flow for consistent performance

### Module Structure

- **`src/poker.zig`**: Core types (Card, Hand, HandRank) and evaluation logic with comprehensive tests
- **`src/main.zig`**: CLI interface and integration test
- **`src/benchmark.zig`**: Performance testing utilities and torture test cases
- **`src/profiler.zig`**: Advanced profiling system for performance analysis
- **`src/profile_main.zig`**: Dedicated profiling executable entry point

### Key Data Structures

- `Card`: Bit-packed representation using u64 bitfield
- `Hand`: Collection of cards with evaluation methods
- `HandRank`: Enum ordered by poker strength for direct comparison

## Code Conventions

- **Types**: Use explicit enum backing types (`u2`, `u4`) for memory efficiency
- **Naming**: snake_case for variables/functions, PascalCase for types/enums
- **Imports**: `std` first, then local modules (`const poker = @import("poker.zig")`)
- **Performance**: Use `inline for` for critical paths, `@intCast` for safe type conversions
- **Testing**: Tests are embedded in `main.zig`, use `parseCards()` for readable test data

## Performance Considerations

- Cards are represented as single bits in a u64 bitfield (52 cards + room for expansion)
- Rank counting uses parallel bit operations across all 13 ranks simultaneously
- Suit extraction uses pre-computed bit masks (0x1111..., 0x2222..., etc.) for parallel operations
- Straight detection uses bit mask shifting instead of sequential checking
- Critical functions marked `inline` for zero-cost calls
- No large lookup tables or memory allocations in hot paths

## Optimization Research

This project includes extensive performance optimization research:

- **`EXPERIMENTS.md`**: Cutting-edge optimization experiments with Apple M1/ARM64 focus
- Documents both successful and failed optimizations with detailed analysis
- Includes benchmarking frameworks and optimization roadmaps

## Profiling System

The project includes a comprehensive profiling system for performance analysis:

### Quick Profiling
```bash
# Run all profiling suites
zig build profile -Doptimize=ReleaseFast
```

### Profiling Components

1. **Component-Level Profiling**: Breaks down evaluation into individual components
   - Full evaluation timing
   - Rank extraction performance
   - Flush detection performance
   - Straight detection performance
   - Pair counting performance

2. **Instruction-Level Analysis**: Compares optimization approaches
   - Tests different rank extraction methods
   - Measures instruction-level performance differences
   - Apple Silicon specific optimizations

3. **Memory Access Pattern Analysis**: Tests cache behavior
   - Sequential vs random access patterns
   - Cache penalty measurement
   - Memory bandwidth utilization

### Profiling Results Format
```
Function                       Calls   Total (ms)   Avg (ns)   Min (ns)   Max (ns)
--------------------------------------------------------------------------------
full_evaluation               100000        1.555       15.6          0       1000
rank_extraction               100000        1.634       16.3          0       1000
flush_detection               100000        1.609       16.1          0       2000
```

### Advanced Profiling (Optional)

For detailed CPU profiling on macOS:
```bash
# Build profiling executable
zig build profile -Doptimize=ReleaseFast

# Use macOS Instruments (if available)
instruments -t "Time Profiler" ./zig-out/bin/profile
instruments -t "Allocations" ./zig-out/bin/profile
```

### When making performance changes:
1. **Always profile**: Use `zig build profile -Doptimize=ReleaseFast` before and after changes
2. **Document results**: Update both `README.md` performance table and `EXPERIMENTS.md`
3. **Test correctness**: Ensure optimizations don't break hand evaluation accuracy
4. **Compare components**: Look at component-level changes to identify bottlenecks
5. **Target Apple M1**: Primary development/testing platform is ARM64