# AGENT.md - Zig Poker Hand Evaluator

**CRITICAL**: Runtime performance is absolutely critical. Use `-Doptimize=ReleaseFast` for real speed testing.
See [DESIGN.md](DESIGN.md) for complete design specification.

**IMPORTANT**: If you cannot implement the exact design specified, DO NOT fall back to simpler solutions. The design must be followed precisely to achieve 2-5ns performance targets.

## Build Commands (Zig 0.14.0)
- `zig build` - Build main executable
- `zig build run` - Run main poker evaluator
- `zig build test` - Run all unit tests
- `zig build bench -Doptimize=ReleaseFast` - Run performance benchmark (target: 2-5ns/hand)
- `zig build build-tables -Doptimize=ReleaseFast` - Generate lookup tables (manual use only, creates src/tables.zig)

## Architecture
High-performance 7-card poker hand evaluator using SIMD and perfect hashing:
- **Main components**: `evaluator.zig` (scalar), `simd_evaluator.zig` (SIMD batched), `chd.zig` (perfect hash)
- **Table generation**: `build_tables.zig` builds perfect hash tables, outputs to `tables.zig` (pre-compiled)
- **Memory footprint**: 120KB total (16KB bucket descriptors + 96KB RankPattern + 8KB FlushRank)
- **Target performance**: 2-5ns per hand, 450M hands/sec single-thread on 3.5GHz CPU

## Code Style
- **Zig version**: 0.14.0 syntax (print requires `.{}` parameter: `print("text", .{});`)
- **Naming**: snake_case for functions/variables, PascalCase for types
- **Imports**: `const std = @import("std");` first, then local imports
- **Error handling**: Use Zig's error unions `!T`, propagate with `try`
- **SIMD**: Use `@Vector(16, u64)` for AVX-512, fallback to 8-lane AVX2
- **Testing**: Unit tests in each file, comprehensive tests in `test_evaluator.zig`
- **Constants**: `const` for lookup tables, prefer compile-time known values
