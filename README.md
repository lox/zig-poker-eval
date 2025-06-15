# Zig Poker Hand Evaluator

A high-performance 7-card Texas Hold'em hand evaluator written in Zig.

## Setup

Uses [Hermit](https://github.com/cashapp/hermit) for dependency management:

```bash
. bin/activate-hermit  # Activates Zig environment
```

## Usage

```bash
# Run optimized benchmarks (production performance)
zig build run -Doptimize=ReleaseFast

# Run tests
zig build test

# Run comprehensive performance profiling
zig build profile -Doptimize=ReleaseFast
```

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

## Performance

Benchmarked on an Apple Macbook Air M1.

```bash
zig build run -Doptimize=ReleaseFast
=== Zig 7-Card Texas Hold'em Evaluator ===
=== Benchmark  ===
Generating 10000 random hands...
Run 1: 52080000 ops, 19.00 ns/op
Run 2: 54750000 ops, 18.00 ns/op
Run 3: 55780000 ops, 17.00 ns/op

=== Performance Summary ===
18.00 ns/op (average across 3 runs)
55.6M evaluations/second
```

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

For detailed optimization techniques and experimental results, see `EXPERIMENTS.md`.

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