<!-- Generated: 2025-07-09 10:32:15 UTC -->

# Development Guide

This guide covers the development environment, code style, common patterns, and workflows for the Zig poker evaluator project.

## Development Environment

### Prerequisites

- **Zig Version**: 0.14.0 (required)
- **Architecture**: Optimized for x86-64 (AVX-512/AVX2) and ARM64 (Apple M1/M2)
- **Build System**: Zig build system with module dependencies

### Project Structure

```text
zig-poker-eval/
├── src/
│   ├── card.zig              # Core card representation (u64 bitfields)
│   ├── evaluator.zig         # Main evaluation engine (scalar + SIMD)
│   ├── hand.zig              # Hand parsing and combination generation
│   ├── range.zig             # Range notation parsing (AA, AKs, etc.)
│   ├── equity.zig            # Equity calculation engine
│   ├── analysis.zig          # Hand analysis utilities
│   ├── draws.zig             # Draw detection and outs
│   ├── poker.zig             # Main public API module
│   ├── cli/                  # CLI implementation
│   │   ├── main.zig          # Entry point
│   │   ├── cli.zig           # Generic CLI framework
│   │   └── ansi.zig          # Terminal colors
│   ├── internal/             # Internal implementation details
│   │   ├── tables.zig        # Pre-compiled lookup tables
│   │   ├── build_tables.zig  # Table generation
│   │   ├── slow_evaluator.zig # Reference implementation
│   │   └── mphf.zig          # Perfect hash functions
│   └── tools/                # Development tools
│       ├── benchmark.zig      # Performance benchmarking
│       └── test_runner.zig    # Test utilities
├── docs/                      # Documentation
└── build.zig                  # Build configuration
```

## Code Style

### Naming Conventions

The project follows idiomatic Zig naming conventions:

```zig
// Types: PascalCase
pub const HandRank = u16;
pub const HandCategory = enum(u4) { ... };

// Functions: camelCase
pub fn evaluateHand(hand: card.Hand) HandRank { ... }
pub fn getHandCategory(rank: HandRank) HandCategory { ... }

// Variables and parameters: snake_case
var rank_counts = [_]u8{0} ** 13;
const suit_mask = @as(u16, @truncate(hand >> offset));

// Constants: UPPER_SNAKE_CASE or snake_case
const RANK_MASK = 0x1FFF;
const DEFAULT_BATCH_SIZE = 32;

// Module-level constants often use snake_case
const table_size = 131072;
```

### Import Organization

Imports follow a consistent pattern:

```zig
// 1. Standard library first
const std = @import("std");

// 2. Project modules
const card = @import("card");
const evaluator = @import("evaluator");

// 3. Internal dependencies (with comment)
// Internal dependencies
const tables = @import("internal/tables.zig");
const slow_eval = @import("internal/slow_evaluator.zig");

// 4. Re-export common types for convenience
const testing = std.testing;
```

### Error Handling

Use Zig's error unions and propagate with `try`:

```zig
// Define specific errors
pub const ParseError = error{
    InvalidCardString,
    InvalidRank,
    InvalidSuit,
    InvalidNotation,
};

// Return error unions
pub fn maybeParseCard(card_str: []const u8) !Hand {
    if (card_str.len != 2) return error.InvalidCardString;
    // ...
}

// Propagate errors with try
const card = try maybeParseCard("As");
```

### Testing Patterns

Each module includes comprehensive tests:

```zig
// Group related tests with descriptive names
test "hand category conversion" {
    try testing.expect(getHandCategory(1) == .straight_flush);
    try testing.expect(getHandCategory(100) == .four_of_a_kind);
}

// Use specific test data with clear descriptions
test "two trips edge cases" {
    const test_cases = [_]struct { hand: u64, desc: []const u8 }{
        .{ .hand = aaakkk8, .desc = "AAA KKK 8" },
        .{ .hand = trips_222_333_a, .desc = "222 333 A" },
    };

    for (test_cases) |tc| {
        const result = evaluateHand(tc.hand);
        // Only print on failure to keep tests clean
        if (result != expected) {
            std.debug.print("FAIL ({s}): {}\n", .{ tc.desc, result });
        }
    }
}

// Ensure all module tests are discovered
test {
    std.testing.refAllDecls(@This());
}
```

### Print Statements

Zig 0.14.0 requires empty parameters for print statements:

```zig
// Correct in 0.14.0
std.debug.print("Hello\n", .{});
std.debug.print("Value: {}\n", .{value});

// Incorrect (will not compile)
std.debug.print("Hello\n");  // Missing .{}
```

## Common Patterns

### 1. Bitfield Card Representation

The core uses a 64-bit packed format for maximum performance:

```zig
/// Layout: [13 spades][13 hearts][13 diamonds][13 clubs]
pub const Hand = u64;

// Create cards with bit shifts
pub fn makeCard(suit: Suit, rank: Rank) Hand {
    const suit_num: u8 = @intFromEnum(suit);
    const rank_num: u8 = @intFromEnum(rank);
    const offset = suit_num * 13;
    return @as(Hand, 1) << @intCast(offset + rank_num);
}

// Extract suit masks efficiently
pub fn getSuitMask(hand: Hand, suit: Suit) u16 {
    const suit_num: u8 = @intFromEnum(suit);
    const offset: u6 = @intCast(suit_num * 13);
    return @as(u16, @truncate((hand >> offset) & RANK_MASK));
}
```

### 2. Compile-Time String Parsing

Parse poker notation at compile time for zero runtime cost:

```zig
/// Parse at compile time - errors are compile errors
pub fn parseCard(comptime card_str: []const u8) Hand {
    if (card_str.len != 2) {
        @compileError("Card must be exactly 2 characters: " ++ card_str);
    }
    // ... parsing logic ...
}

// Usage - resolved at compile time
const ace_spades = parseCard("As");
const test_hand = parseCard("Ac") | parseCard("Kc") | parseCard("Qc");
```

### 3. SIMD Patterns

Architecture-adaptive SIMD with compile-time dispatch:

```zig
fn computeRpcSimd(comptime batchSize: usize, hands: *const [batchSize]u64) [batchSize]u32 {
    // Compile-time dispatch based on batch size
    if (batchSize == 2 or batchSize == 4 or batchSize == 8 or batchSize == 16) {
        // SIMD path using @Vector
        const clubs_v: @Vector(batchSize, u16) = clubs;
        const diamonds_v: @Vector(batchSize, u16) = diamonds;

        // Vectorized operations
        const rank_bit: @Vector(batchSize, u16) = @splat(@as(u16, 1) << @intCast(rank));
        const has_rank = @select(u8, (clubs_v & rank_bit) != zero_vec, one_vec, zero_u8_vec);

        // Base-5 encoding with SIMD multiply
        const five_vec: @Vector(batchSize, u32) = @splat(5);
        rpc_vec = rpc_vec * five_vec + rank_count_vec;
    } else {
        // Scalar fallback for odd batch sizes
        for (hands, 0..) |hand, i| {
            result[i] = computeRpcFromHand(hand);
        }
    }
}
```

### 4. Module Dependency Pattern

Modules are organized in dependency layers:

```zig
// build.zig - define module dependencies
const card_mod = b.addModule("card", .{
    .root_source_file = b.path("src/card.zig"),
});

const evaluator_mod = b.addModule("evaluator", .{
    .root_source_file = b.path("src/evaluator.zig"),
});
evaluator_mod.addImport("card", card_mod);

// In evaluator.zig
const card = @import("card");  // Use the imported module
```

### 5. Memory Management Pattern

Use allocators explicitly for dynamic memory:

```zig
pub fn generateSuitedCombinations(rank1: Rank, rank2: Rank, allocator: std.mem.Allocator) ![]const [2]Hand {
    var combinations = try allocator.alloc([2]Hand, 4);
    defer allocator.free(combinations);  // In calling code

    // Fill combinations...
    return combinations;
}

// Caller manages memory
const combinations = try generateSuitedCombinations(.ace, .king, allocator);
defer allocator.free(combinations);
```

### 6. Generic CLI Framework

The CLI uses comptime reflection for automatic command parsing:

```zig
// Define commands as a tagged union
pub const Commands = union(enum) {
    eval: EvalCommand,
    range: RangeCommand,
    bench: BenchCommand,

    pub const EvalCommand = struct {
        hand: []const u8,
        verbose: bool = false,
    };
};

// The framework automatically parses based on struct fields
const result = try Cli(Commands).parseArgs(allocator, args);
switch (result.command) {
    .eval => |cmd| { /* handle eval */ },
    .range => |cmd| { /* handle range */ },
    .bench => |cmd| { /* handle bench */ },
}
```

## Common Workflows

### Building and Running

```bash
# Build main executable
task build

# Run with arguments
task run -- eval "AsKsQsJsTs"

# Build for library use
zig build -Doptimize=ReleaseFast
```

### Testing

```bash
# Run all tests
task test

# Run with detailed output (via zig build)
zig build test --summary all

# Test specific module
zig test src/evaluator.zig
```

### Benchmarking

```bash
# Run performance benchmarks
task bench:eval
task bench:equity
task bench:showdown

# With custom options
task bench:eval -- --iterations 10000000 --validate
```

### Profiling

```bash
# Profile different workloads
task profile:eval
task profile:equity
task profile:showdown

# Analyze results
uniprof analyze /tmp/eval_profile/profile.json
uniprof visualize /tmp/eval_profile/profile.json
```

### Table Generation

The lookup tables are pre-compiled, but can be regenerated:

```bash
# Regenerate tables (takes several minutes)
zig build build-tables -Doptimize=ReleaseFast

# This creates src/internal/tables.zig
```

### Adding New Features

1. **Add new module**: Create in `src/`, follow naming conventions
2. **Update build.zig**: Add module definition with dependencies
3. **Add tests**: Include comprehensive tests in the module
4. **Update poker.zig**: Export public API if needed
5. **Document**: Add usage examples in tests or comments

## Performance Considerations

### Critical Path Optimizations

1. **Bitfield operations**: Use single u64 for card representation
2. **SIMD batching**: Process multiple hands in parallel
3. **Perfect hashing**: O(1) lookup with minimal memory access
4. **Branch-free design**: Avoid conditionals in hot paths

### Memory Layout

```zig
// Align data structures for SIMD
const BatchResult = struct {
    ranks: [32]u16 align(64),  // Cache line aligned
};

// Use packed structures where beneficial
const PackedRank = packed struct {
    rank1: u13,
    rank2: u13,
    _padding: u6,
};
```

### Profiling Integration

```bash
# Use task-based profiling with uniprof
task profile:eval
task profile:equity
task profile:showdown

# Analyze results
uniprof analyze /tmp/eval_profile/profile.json
uniprof visualize /tmp/eval_profile/profile.json
```

## Reference

### File Organization

- `src/*.zig` - Core library modules
- `src/internal/*.zig` - Implementation details (not public API)
- `src/cli/*.zig` - Command-line interface
- `src/tools/*.zig` - Development and benchmarking tools
- `docs/*.md` - Documentation

### Naming Conventions Summary

- **Types**: PascalCase (`HandRank`, `CardSet`)
- **Functions**: camelCase (`evaluateHand`, `parseCard`)
- **Variables**: snake_case (`rank_counts`, `suit_mask`)
- **Constants**: UPPER_SNAKE_CASE or snake_case (`RANK_MASK`)
- **Modules**: lowercase (`card`, `evaluator`)
- **Test names**: descriptive strings with spaces

### Common Imports

```zig
// In most modules
const std = @import("std");
const testing = std.testing;

// Core types available everywhere
const card = @import("card");
pub const Hand = card.Hand;
pub const Suit = card.Suit;
pub const Rank = card.Rank;
```
