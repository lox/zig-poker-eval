# AGENT.md - Zig Poker Evaluator

High-performance 7-card Texas Hold'em poker hand evaluator. See README.md for detailed information.

## Essential Commands
- **Setup**: `. bin/activate-hermit` (activates Zig environment)
- **Run**: `zig build run -Doptimize=ReleaseFast` (production performance)
- **Test**: `zig build test` 
- **Profile**: `zig build profile -Doptimize=ReleaseFast`

## Code Style
- **Naming**: snake_case for variables/functions, PascalCase for types/enums
- **Performance**: Use `inline for` for critical paths, `@intCast` for safe casts
- **Testing**: Tests embedded in main.zig, use `parseCards()` for readable test data

## Key Implementation Details
- Cards represented as single bits in u64 bitfield
- Enum values match poker strength ordering for direct comparison
- Zero-cost abstractions with compile-time optimization
