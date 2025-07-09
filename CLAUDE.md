# CLAUDE.md - Zig Poker Hand Evaluator

**CRITICAL**: Runtime performance is absolutely critical. Use `-Doptimize=ReleaseFast` for real speed testing.
Achieved performance: ~4.5ns per hand on Apple M1.

## Build Commands (Zig 0.14.0)

- `zig build` - Build main executable
- `zig build run` - Run main poker evaluator
- `zig build test` - Run all unit tests (82 tests across all modules)
- `zig build test --summary all` - Run tests with detailed summary of all modules
- `zig build bench -Doptimize=ReleaseFast` - Run performance benchmark (~4.5ns/hand on M1)
- `zig build build-tables -Doptimize=ReleaseFast` - Generate lookup tables (manual use only, creates src/internal/tables.zig)

## Architecture

High-performance 7-card poker hand evaluator using SIMD batch processing and CHD perfect hash tables:

- **Main components**: `src/evaluator.zig` (single + SIMD batch evaluation), `src/internal/mphf.zig` (CHD perfect hash)
- **Table generation**: `src/internal/build_tables.zig` generates lookup tables, outputs to `src/internal/tables.zig`
- **Memory footprint**: ~267KB total (8KB displacement array + 256KB value table + 3KB flush table)
- **Measured performance**: ~4.5ns per hand, 224M+ hands/sec on Apple M1

See documentation:

- [docs/design.md](docs/design.md) - Implementation architecture and algorithms
- [docs/benchmarking.md](docs/benchmarking.md) - Performance measurement methodology
- [docs/profiling.md](docs/profiling.md) - Profiling tools and techniques
- [docs/experiments.md](docs/experiments.md) - Optimization experiments and learnings

## Code Style

- **Zig version**: 0.14.0 syntax (print requires `.{}` parameter: `print("text", .{});`)
- **Naming**: camelCase for functions, snake_case for variables, PascalCase for types (idiomatic Zig)
- **Imports**: `const std = @import("std");` first, then local imports
- **Error handling**: Use Zig's error unions `!T`, propagate with `try`
- **SIMD**: Use `@Vector(batchSize, T)` with optimal batch size 32 for best performance
- **Testing**: Unit tests in each file using Zig's built-in test framework
- **Constants**: `const` for lookup tables, prefer compile-time known values
