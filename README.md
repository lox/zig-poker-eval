# Zig Poker Hand Evaluator

A high-performance 7-card Texas Hold'em hand evaluator written in Zig, achieving **48+ million evaluations per second** in realistic scenarios using advanced lookup table optimization and optimized flush detection.

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

# Run comprehensive performance profiling
zig build profile -Doptimize=ReleaseFast
```

## Performance

**Benchmarked on Apple M1 (apple_a14)**

| Test Type | Build Mode | Hands/Second | Nanoseconds/Hand |
|-----------|------------|--------------|------------------|
| Realistic (10M unique hands) | Debug | ~7.7M | 130ns |
| Realistic (10M unique hands) | ReleaseFast | **~48M** | **21ns** |

*Realistic benchmark uses 10M unique random hands with memory pressure to simulate real-world usage patterns.*

## Design

This evaluator uses advanced optimization techniques for maximum performance:

- **Optimized flush detection**: Parallel suit extraction with bit manipulation eliminates nested loops
- **Rank Distribution LUT**: 64-byte lookup table for instant non-flush hand categorization
- **Cards as bits**: Each card represented as a single bit in a u64 bitfield
- **Compile-time optimization**: Lookup tables generated at build time using Zig's `comptime`
- **CPU-native operations**: Leverages @popCount and inline loops for maximum performance
- **Cache-friendly**: Minimal memory footprint with efficient data structures

### Architecture

- **`poker.zig`**: Core evaluation logic, data structures, and comprehensive tests
- **`benchmark.zig`**: Performance testing and random hand generation
- **`profiler.zig`**: Advanced profiling system for performance analysis
- **`main.zig`**: CLI interface and integration test

## Profiling

### Quick Profiling
```bash
# Run comprehensive performance analysis
zig build profile -Doptimize=ReleaseFast
```

### Profiling Output
The profiler provides detailed component-level analysis:

```
Function                       Calls   Total (ms)   Avg (ns)   Min (ns)   Max (ns)
--------------------------------------------------------------------------------
full_evaluation               100000        1.555       15.6          0       1000
rank_extraction               100000        1.634       16.3          0       1000
flush_detection               100000        1.609       16.1          0       2000
straight_detection            100000        1.575       15.8          0       1000
pair_counting                 100000        1.590       15.9          0       2000
```

### Profiling Features
- **Component-level timing**: Individual function performance analysis
- **Statistical analysis**: Min/max/average timing with variance detection
- **Instruction-level comparison**: Different optimization approach comparisons
- **Memory access patterns**: Cache behavior and memory bandwidth analysis
- **Apple Silicon optimization**: M1/M2/M3 specific performance insights

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
├── main.zig         # Entry point and CLI interface
├── poker.zig        # Core poker types, evaluation logic, and tests
├── benchmark.zig    # Performance testing utilities
├── profiler.zig     # Advanced profiling system
└── profile_main.zig # Profiling executable entry point

CLAUDE.md         # AI coding assistant instructions
EXPERIMENTS.md    # Detailed optimization research and results
build.zig         # Build configuration
bin/              # Hermit-managed Zig installation
```