<!-- Generated: 2025-01-09 18:45:00 UTC -->

# Deployment Guide

This guide covers packaging and distributing the Zig Poker Evaluator as both a library and command-line tool.

## Overview

The Zig Poker Evaluator can be deployed in two primary forms:

1. **Library Package**: For integration into other Zig projects via the package manager
2. **CLI Executable**: As a standalone command-line tool for poker analysis

The build system supports cross-platform compilation and multiple optimization levels, with `-Doptimize=ReleaseFast` recommended for production deployments to achieve the 2-5ns per hand performance target.

## Package Types

### Library Deployment

The poker evaluator exposes a modular API through the main `poker` module:

```zig
// In your build.zig.zon
.dependencies = .{
    .zig_poker_eval = .{
        .url = "https://github.com/USERNAME/zig-poker-eval/archive/v1.0.0.tar.gz",
        .hash = "1220...", // Use `zig fetch` to get the hash
    },
},

// In your build.zig
const poker_dep = b.dependency("zig_poker_eval", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("poker", poker_dep.module("poker"));
```

The library provides these core modules:
- `poker` - Main API module combining all functionality
- `card` - Card representation and manipulation
- `evaluator` - Core hand evaluation engine
- `equity` - Monte Carlo and exact equity calculations
- `range` - Range notation parsing and analysis
- `analysis` - Hand analysis utilities
- `draws` - Draw detection and outs calculation

### Executable Deployment

Build the CLI tool for distribution:

```bash
# Build optimized release binary
zig build -Doptimize=ReleaseFast

# Output location: zig-out/bin/poker-eval
```

The CLI provides these commands:
- `eval` - Evaluate 7-card poker hands
- `equity` - Calculate hand vs hand equity
- `range` - Parse and analyze hand ranges
- `bench` - Run performance benchmarks
- `demo` - Interactive demonstration

## Platform Deployment

### Build Commands by Platform

```bash
# macOS (Intel)
zig build -Dtarget=x86_64-macos -Doptimize=ReleaseFast

# macOS (Apple Silicon)
zig build -Dtarget=aarch64-macos -Doptimize=ReleaseFast

# Linux x86_64
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseFast

# Windows x64
zig build -Dtarget=x86_64-windows -Doptimize=ReleaseFast
```

### Platform-Specific Considerations

**macOS**:
- Universal binary support via `lipo` for combined Intel/ARM distribution
- Code signing may be required for distribution outside App Store
- Minimum macOS version: 10.15 (Catalina)

**Linux**:
- Static linking recommended for maximum portability
- SIMD support: AVX2 minimum, AVX-512 for optimal performance
- Consider AppImage or Flatpak for desktop distribution

**Windows**:
- Ships as single `.exe` with no dependencies
- MSVC runtime not required (Zig uses its own libc)
- Consider code signing for SmartScreen compatibility

### Performance Optimization Flags

```bash
# Maximum performance (recommended for production)
zig build -Doptimize=ReleaseFast

# Balanced performance with safety checks
zig build -Doptimize=ReleaseSafe

# Small binary size (trades some performance)
zig build -Doptimize=ReleaseSmall
```

## Reference

### Output Locations

```
zig-out/
├── bin/
│   ├── poker-eval          # Main CLI executable
│   ├── generate-all-hands  # Test data generator
│   └── verify-all-hands    # Verification tool
└── lib/                    # (When building as library)
```

### Build Configuration

Key build options in `build.zig`:
- Target architecture: `-Dtarget=<triple>`
- Optimization mode: `-Doptimize=<mode>`
- Custom allocator: Configurable via CLI for different memory patterns

### Release Checklist

1. **Version Update**: Update version in `build.zig.zon`
2. **Performance Validation**: Run benchmarks to verify 2-5ns target
   ```bash
   zig build bench -Doptimize=ReleaseFast
   ```
3. **Cross-Platform Testing**: Build and test on all target platforms
4. **Correctness Verification**: Run full test suite
   ```bash
   zig build test
   zig build verify-all -Doptimize=ReleaseFast
   ```
5. **Package Hash**: Generate hash for Zig package manager
   ```bash
   zig fetch --save=zig_poker_eval file://path/to/release.tar.gz
   ```

### Distribution Formats

**Library Package**:
- GitHub releases with source tarball
- Include generated `tables.zig` for zero-config usage
- Semantic versioning (currently v1.0.0)

**CLI Distribution**:
- Platform-specific binaries in GitHub releases
- Homebrew formula for macOS
- AUR package for Arch Linux
- Scoop manifest for Windows

### Memory and Performance

The evaluator has minimal memory requirements:
- **Lookup tables**: 120KB total (compiled into binary)
- **Runtime memory**: ~1KB stack usage per evaluation
- **Batch processing**: Scales linearly with batch size

Performance characteristics:
- Single evaluation: 2-5ns per hand
- Batch evaluation: 450M+ hands/second (single thread)
- Monte Carlo: 50M+ simulations/second
- Memory bandwidth: ~2.4GB/s at peak throughput
