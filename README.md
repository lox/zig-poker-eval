# Zig Poker Hand Evaluator

A high-performance 7-card Texas Hold'em hand evaluator written in Zig, achieving **34+ million evaluations per second**.

## Setup

Uses [Hermit](https://github.com/cashapp/hermit) for dependency management:

```bash
. bin/activate-hermit  # Activates Zig environment
```

## Usage

```bash
# Run the demo and benchmarks
zig build run

# Run optimized benchmarks (production performance)
zig build run -Doptimize=ReleaseFast

# Run tests
zig build test
```

## Performance

**Benchmarked on Apple M1 (apple_a14)**

| Test Type | Build Mode | Hands/Second | Nanoseconds/Hand |
|-----------|------------|--------------|------------------|
| Random hands | Debug | ~3.6M | 277ns |
| Random hands | ReleaseFast | **~34M** | **29ns** |
| Torture cases | ReleaseFast | **~39M** | **26ns** |

## Design

### High-Performance Bit Manipulation
- **Cards as bits**: Each card represented as a single bit in a u64
- **Inline loops**: Critical paths use `inline for` for zero-cost iteration
- **@popCount**: Leverages CPU's native population count for rank counting
- **Optimized straight detection**: Bit mask shifting instead of sequential checking

### Architecture
- **`poker.zig`**: Core types (Card, Hand, HandRank) and evaluation logic
- **`benchmark.zig`**: Random hand generation and torture test cases
- **`main.zig`**: CLI interface, demos, and comprehensive tests

### Key Optimizations
1. **Bit-packed cards**: 52 cards fit in a single u64 with room to spare
2. **Parallel rank counting**: Process all 13 ranks simultaneously using bit shifts
3. **Cache-friendly**: No large lookup tables, minimal memory allocations
4. **Branchless evaluation**: Optimized control flow for consistent performance

## Example

```zig
const poker = @import("poker.zig");

// Create a 7-card hand (hole cards + community cards)
const hand = poker.createHand(&.{
    .{ .hearts, .ace },   // Hole card 1
    .{ .spades, .ace },   // Hole card 2
    .{ .hearts, .king },  // Flop
    .{ .hearts, .queen }, // Flop
    .{ .hearts, .jack },  // Flop
    .{ .hearts, .ten },   // Turn
    .{ .clubs, .two },    // River
});

const rank = hand.evaluate(); // .straight_flush (royal flush)
```

## Project Structure

```
src/
├── main.zig      # Entry point, CLI, and tests
├── poker.zig     # Core poker types and evaluation logic
└── benchmark.zig # Performance testing utilities

build.zig         # Build configuration
bin/              # Hermit-managed Zig installation
```