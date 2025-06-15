# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Test Commands

```bash
# Build and run optimized (production performance)
zig build run -Doptimize=ReleaseFast

# Run tests
zig build test

# Run comprehensive performance profiling
zig build profile -Doptimize=ReleaseFast
```

## Architecture Overview

High-performance 7-card Texas Hold'em poker hand evaluator using bit manipulation and optimized algorithms. See README.md for detailed performance metrics and design information.

### Module Structure

- **`src/poker.zig`**: Core evaluation logic and data structures
- **`src/main.zig`**: CLI interface and benchmarking
- **`src/benchmark.zig`**: Performance testing utilities
- **`src/profiler.zig`**: Advanced profiling system

## Code Conventions

- **Naming**: snake_case for variables/functions, PascalCase for types/enums
- **Performance**: Use `inline for` for critical paths, `@intCast` for safe type conversions
- **Testing**: Tests embedded in `main.zig`, use `parseCards()` for readable test data

## Performance Optimization

When making performance changes:
1. **Profile before/after**: Use `zig build profile -Doptimize=ReleaseFast`
2. **Test correctness**: Ensure optimizations don't break evaluation accuracy
3. **Target Apple M1**: Primary development/testing platform is ARM64