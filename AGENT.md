# AGENT.md - Zig Poker Evaluator

## Commands
- **Build/Run**: `zig build run` (debug), `zig build run -Doptimize=ReleaseFast` (production)
- **Test**: `zig build test` (runs all tests in main.zig)
- **Environment**: `. bin/activate-hermit` (activates Zig environment)

## Architecture
- **High-performance 7-card Texas Hold'em evaluator** achieving 34M+ evals/sec
- **Core modules**: `poker.zig` (evaluation logic), `main.zig` (CLI/tests), `benchmark.zig` (perf testing)
- **Bit manipulation design**: Cards as bits in u64, leverages @popCount and inline loops
- **No external dependencies**: Pure Zig implementation with zero-cost abstractions

## Code Style
- **Naming**: snake_case for variables/functions, PascalCase for types/enums
- **Types**: Explicit enum backing types (u2, u4), bit manipulation with u64
- **Imports**: std import first, then local modules (`const poker = @import("poker.zig")`)
- **Performance**: Use `inline for` for critical paths, @intCast for safe casts
- **Testing**: Tests embedded in main.zig, use parseCards() for readable test data
- **Documentation**: Concise comments for complex bit operations only
- **Error handling**: Use try/catch for allocations, comptime known operations preferred

## Key Patterns
- Cards represented as single bits in u64 bitfield
- Enum values match poker strength ordering for direct comparison
