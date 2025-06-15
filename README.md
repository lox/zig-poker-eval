# Zig Poker Hand Evaluator

A high-performance 7-card Texas Hold'em hand evaluator written in Zig, achieving **63+ million evaluations per second**.

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
| Random hands | ReleaseFast | **~63M** | **16ns** |
| Torture cases | ReleaseFast | **~69M** | **14ns** |

## Design

This evaluator uses efficient bit manipulation techniques optimized for modern CPUs:

- **Cards as bits**: Each card represented as a single bit in a u64 bitfield
- **Compile-time optimization**: Lookup tables generated at build time
- **CPU-native operations**: Leverages @popCount and inline loops for maximum performance
- **Cache-friendly**: Minimal memory footprint with efficient data structures

### Architecture

- **`poker.zig`**: Core evaluation logic and data structures
- **`benchmark.zig`**: Performance testing and random hand generation  
- **`main.zig`**: CLI interface and comprehensive test suite

For detailed optimization techniques and experimental results, see `EXPERIMENTS.md`.

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