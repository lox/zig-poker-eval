<!-- Generated: 2025-07-09 07:24:34 UTC -->

# Zig Poker Evaluator Build System

## Overview

The Zig poker evaluator uses Zig's built-in build system (defined in `build.zig`) to manage compilation, testing, benchmarking, and tooling. The build system is designed around a modular architecture with clear dependency hierarchy and supports multiple optimization levels for performance-critical components.

### Build Configuration Files

- **`build.zig`** - Main build script defining modules, dependencies, and build targets
- **`build.zig.zon`** - Build manifest specifying project metadata and Zig version requirements

### Module Architecture

The build system defines a hierarchical module structure (lines 7-63 in `build.zig`):

1. **Level 1: Core Foundation**
   - `card` module (line 10) - Card representation, no dependencies

2. **Level 2: Evaluation Engine**
   - `evaluator` module (line 15) - Hand evaluation algorithms, depends on card

3. **Level 3: Game Logic**
   - `hand` module (line 21) - Hand combinations, depends on card
   - `equity` module (line 27) - Equity calculations, depends on card and evaluator

4. **Level 4: Advanced Features**
   - `range` module (line 34) - Hand ranges, depends on card, hand, and equity
   - `analysis` module (line 42) - Statistical analysis, depends on card
   - `draws` module (line 48) - Draw detection, depends on card

5. **Level 5: Main API**
   - `poker` module (line 54) - Public API aggregating all modules

## Build Workflows

### Task Automation (Recommended)

The project uses [Task](https://taskfile.dev) for common workflows:

```bash
# Build main executable
task build

# Run main executable
task run -- <args>

# Run all tests
task test

# Run benchmarks
task bench:eval
task bench:equity
task bench:showdown

# Profile performance
task profile:eval
task profile:equity
task profile:showdown
```

See `Taskfile.yml` for all available tasks.

### Direct Zig Build Commands

For advanced use cases or build system development:

```bash
# Basic build
zig build

# Run main executable
zig build run

# Run all tests
zig build test

# Run tests with detailed output
zig build test --summary all

# Generate lookup tables (manual use only)
zig build build-tables -Doptimize=ReleaseFast

# Generate all hand evaluations (133M hands)
zig build gen-all

# Verify evaluator correctness
zig build verify-all
```

### Build Configurations

The build system uses standard Zig optimization levels:

- **Debug** (default) - Full debug info, runtime safety checks
- **ReleaseSafe** - Optimized with safety checks
- **ReleaseFast** - Maximum performance, no safety checks (required for benchmarks)
- **ReleaseSmall** - Optimized for size

Target platform can be specified with `-Dtarget=`:
```bash
zig build -Dtarget=x86_64-macos -Doptimize=ReleaseFast
```

## Platform Setup Requirements

### Zig Version
- **Minimum Version**: 0.14.0 (specified in `build.zig.zon` line 4)
- **Syntax Requirements**: Uses Zig 0.14.0 print syntax with `.{}` parameter

### File Paths and Artifacts

Build outputs are organized as follows:
- **Executables**: `zig-out/bin/`
  - `poker-eval` - Main CLI executable
  - `build_tables` - Table generator
  - `generate-all-hands` - Test data generator
  - `verify-all-hands` - Verification tool
- **Build Cache**: `zig-cache/` (auto-generated)
- **Source**: `src/` with modular organization

### Dependencies

The project has no external dependencies (line 7 in `build.zig.zon`). All functionality is implemented in pure Zig.

## Reference

### Build Targets

| Target | Command | Description | Configuration |
|--------|---------|-------------|---------------|
| **default** | `zig build` | Build main executable | Lines 93-103 |
| **run** | `zig build run` | Run poker-eval CLI | Lines 115-122 |
| **test** | `zig build test` | Run all unit tests | Lines 153-243 |
| **bench** | `zig build bench` | Performance benchmark via CLI | Lines 106-112 |
| **build-tables** | `zig build build-tables` | Generate lookup tables | Lines 72-90 |
| **gen-all** | `zig build gen-all` | Generate 133M test hands | Lines 125-136 |
| **verify-all** | `zig build verify-all` | Verify against all hands | Lines 139-150 |

### Optimization Flags

Performance-critical builds should use:
```bash
-Doptimize=ReleaseFast
```

This is required for:
- Benchmarking (target: 2-5ns per hand evaluation)
- Table generation (CPU-intensive perfect hash computation)
- Full verification runs (processing 133M hands)

### Module Import Structure

The build system establishes clear import dependencies:

```zig
// Example from evaluator module setup (lines 15-18)
const evaluator_mod = b.addModule("evaluator", .{
    .root_source_file = b.path("src/evaluator.zig"),
});
evaluator_mod.addImport("card", card_mod);
```

### Troubleshooting

#### Common Issues

1. **Missing Zig Version**
   - Error: `error: minimum Zig version 0.14.0 required`
   - Solution: Update to Zig 0.14.0 or later

2. **Slow Performance**
   - Issue: Benchmark shows >100ns per hand
   - Solution: Ensure using `-Doptimize=ReleaseFast`

3. **Module Import Errors**
   - Issue: `error: no module named 'card' available`
   - Cause: Module dependencies not properly configured
   - Check: Module import declarations in build.zig

4. **Build Cache Issues**
   - Solution: Clean cache with `rm -rf zig-cache zig-out`

#### Build System Internals

Key configuration points in `build.zig`:

- **Module definitions**: Lines 10-63
- **Executable configurations**: Lines 72-103
- **Test configurations**: Lines 153-243
- **Tool configurations**: Lines 125-150

The build system uses Zig's module system (introduced in 0.11.0) for clean dependency management, replacing the older package system.
