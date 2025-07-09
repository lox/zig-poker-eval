# File Catalog - Zig Poker Evaluator

<!-- Generated: 2025-07-09 07:24:23 UTC -->

## Overview

This project implements a high-performance 7-card poker hand evaluator in Zig, achieving 2-5 nanosecond evaluation times through careful use of SIMD instructions and perfect hash tables. The codebase is organized into a clean modular structure with core evaluation logic, supporting utilities, build tools, and comprehensive documentation.

The project architecture separates public API components in the root `src/` directory from internal implementation details in `src/internal/`. The build system includes sophisticated table generation tools that create perfect hash functions for poker hand lookups, while the tools directory contains benchmarking and verification utilities to ensure correctness and performance. Command-line interfaces are isolated in `src/cli/` for clean separation of concerns.

## Core Source Files

The main evaluation engine and public API:

- `src/poker.zig` - **Main entry point**: Public API aggregating all poker functionality
- `src/evaluator.zig` - Core hand evaluation logic using perfect hash tables
- `src/card.zig` - Card representation and manipulation (ranks, suits, deck operations)
- `src/hand.zig` - Hand types, rankings, and comparison logic
- `src/equity.zig` - Monte Carlo equity calculation for hand vs hand/range scenarios
- `src/range.zig` - Poker range parsing and representation (e.g., "AK+", "JJ-99")
- `src/analysis.zig` - Hand analysis utilities (outs, draws, board texture)
- `src/draws.zig` - Draw detection and classification (flushes, straights, etc.)

## Internal Implementation

Low-level implementation details and algorithms:

- `src/internal/tables.zig` - Pre-generated lookup tables (120KB total footprint)
- `src/internal/mphf.zig` - Minimal perfect hash function implementation
- `src/internal/build_tables.zig` - Table generation logic (run via `zig build build-tables`)
- `src/internal/slow_evaluator.zig` - Reference implementation for verification
- `src/internal/simulation.zig` - Monte Carlo simulation engine internals
- `src/internal/notation.zig` - Card notation parsing and formatting
- `src/internal/profile_main.zig` - Profiling harness for performance analysis

## Build System

Build configuration and package management:

- `build.zig` - **Main build script**: Defines all build targets and options
- `build.zig.zon` - Package manifest with dependencies and metadata
- `.github/workflows/` - CI/CD pipeline configurations (if present)

## Tools and Utilities

Development and verification tools:

- `src/tools/benchmark.zig` - Performance benchmarking suite (target: 450M hands/sec)
- `src/tools/test_runner.zig` - Custom test runner with enhanced reporting
- `src/tools/generate_all_hands.zig` - Generates test data for all possible 7-card hands
- `src/tools/verify_all_hands.zig` - Verifies evaluator correctness against all hands
- `scripts/profile.sh` - Shell script for CPU profiling with perf

## Command Line Interface

User-facing CLI components:

- `src/cli/main.zig` - CLI entry point and command dispatch
- `src/cli/cli.zig` - Command implementations (evaluate, equity, analyze)
- `src/cli/ansi.zig` - Terminal color and formatting utilities

## Documentation

Project documentation and design notes:

- `README.md` - Project overview and quick start guide
- `CLAUDE.md` - AI assistant instructions and coding standards
- `docs/design.md` - Detailed architecture and algorithm descriptions
- `docs/benchmarking.md` - Performance measurement methodology
- `docs/profiling.md` - CPU profiling instructions and analysis
- `docs/experiments.md` - Performance experiments and optimizations

## Test Data

Pre-generated test datasets:

- `all_hands.dat` - Binary file containing all 133,784,560 seven-card hands

## Quick Reference

### Key Patterns
- **Public API**: Import `src/poker.zig` for all functionality
- **Card creation**: `Card.fromString("As")` or `Card.init(.ace, .spades)`
- **Hand evaluation**: `evaluator.evaluate7(cards)` returns `HandRank`
- **Equity calculation**: `equity.calculate(hero_range, villain_range, board)`
- **Perfect hashing**: See `mphf.zig` for CHD algorithm implementation

### Build Commands
```bash
zig build                              # Build main executable
zig build test                         # Run all tests (63 tests)
zig build bench -Doptimize=ReleaseFast # Run benchmarks
zig build build-tables                 # Regenerate lookup tables
```

### Performance Targets
- Single hand evaluation: 2-5 nanoseconds
- Throughput: 450M hands/second (single-threaded)
- Memory footprint: 120KB total for all lookup tables

### Key Dependencies
- Zig 0.14.0 or later
- No external dependencies (self-contained)
- SIMD support (AVX2/AVX-512 for optimal performance)
