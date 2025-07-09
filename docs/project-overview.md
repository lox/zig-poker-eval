<!-- Generated: 2025-07-09 07:24:52 UTC -->

# Zig Poker Evaluator - Project Overview

A high-performance 7-card poker hand evaluator and comprehensive analysis toolkit implemented in Zig 0.14.0. This project combines ultra-fast hand evaluation (~4.5ns per hand on Apple M1) with complete poker analysis capabilities including equity calculations, range parsing, and Monte Carlo simulations. The evaluator uses SIMD optimizations and perfect hash tables to achieve 224M+ hands/second throughput on modern CPUs.

The project serves both as a standalone CLI tool for poker analysis and as a library that can be integrated into other Zig projects. It features a modular architecture with clean separation between the core evaluation engine, poker analysis tools, and user interface components.

## Key Files

**Entry Points:**
- `src/cli/main.zig` - Main CLI application with commands for equity, eval, range, bench, and demo
- `src/poker.zig` - Public library API aggregating all poker functionality
- `build.zig` - Build configuration defining module dependencies and build targets

**Core Configuration:**
- `build.zig.zon` - Package manifest requiring Zig 0.14.0 minimum version
- `CLAUDE.md` - Project-specific instructions emphasizing performance targets (2-5ns/hand)
- `bin/hermit` - Hermit environment manager for consistent Zig version

## Technology Stack

**Language & Build:**
- Zig 0.14.0 with specific syntax requirements (e.g., `print("text", .{});`)
- Module-based architecture with explicit dependency graph in `build.zig`
- Hermit for dependency management (`bin/activate-hermit`)

**Performance Technologies:**
- SIMD vectorization using `@Vector(32, u64)` for batch processing (`src/evaluator.zig`)
- Perfect hash tables via CHD algorithm (`src/internal/tables.zig`, 120KB total)
- Cache-optimized data structures with L1-friendly lookup tables

**Algorithms:**
- RPC (Rank Pattern Counting) with base-5 encoding for hand patterns
- Flush detection with optimized bit manipulation
- Monte Carlo simulation for equity calculations (`src/equity.zig`)
- Range notation parser supporting standard poker syntax (`src/range.zig`)

## Platform Support

**Requirements:**
- Zig 0.14.0 (managed via Hermit in `bin/`)
- 64-bit architecture for optimal SIMD performance
- ~120KB memory for lookup tables

**Tested Platforms:**
- macOS (Apple Silicon M1) - Primary development platform
- Linux x86_64 with AVX2/AVX-512 support
- Windows (community supported)

**Build Commands:**
- `zig build` - Build main executable
- `zig build test` - Run 60+ unit tests across all modules
- `zig build bench -Doptimize=ReleaseFast` - Performance benchmark
- `zig build run -- [command]` - Run CLI with specific command
