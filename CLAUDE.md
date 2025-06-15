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
```

## Architecture Overview

This is a high-performance 7-card Texas Hold'em poker hand evaluator achieving 34+ million evaluations per second. The core design uses bit manipulation for maximum performance:

- **Cards as bits**: Each card represented as a single bit in a u64 bitfield
- **Zero-cost abstractions**: Extensive use of `inline for` and `@popCount` for CPU-native operations
- **Branchless evaluation**: Optimized control flow for consistent performance

### Module Structure

- **`src/poker.zig`**: Core types (Card, Hand, HandRank) and evaluation logic
- **`src/main.zig`**: CLI interface, comprehensive tests, and demo code
- **`src/benchmark.zig`**: Performance testing utilities and torture test cases

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
- Straight detection uses bit mask shifting instead of sequential checking
- No large lookup tables or memory allocations in hot paths