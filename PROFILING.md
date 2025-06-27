# Profiling Scripts

Scripts for performance analysis and profiling of the poker hand evaluator.

## Scripts

### `profile.sh`
**Unified profiling script** - High-frequency sampling with configurable parameters.

```bash
./scripts/profile.sh [iterations] [duration] [output_file]
```

**Examples**:
```bash
# Quick profile (default: 20M iterations, 15s sampling)
./scripts/profile.sh

# Custom parameters
./scripts/profile.sh 5000000 10 quick_profile.txt

# Long detailed profile
./scripts/profile.sh 50000000 30 detailed_profile.txt
```

**Features**:
- Builds benchmark with ReleaseFast + debug symbols
- Configurable iteration count (default: 20M)
- Configurable sampling duration (default: 15s) 
- Samples every 1ms for high resolution
- Outputs detailed function-level breakdown
- Error handling for failed sampling

**Use this for identifying performance bottlenecks.**


## Requirements

- **macOS only** (uses `sample` command)
- Benchmark must be built with debug symbols
- For other platforms: Use `perf` (Linux) or VTune (Windows)

## Usage Notes

- Scripts automatically handle argument passing to benchmark
- Use `simple_profile.sh` for most profiling needs
- High iteration counts (5M-20M) provide better sampling data
- Profile output shows line-level hotspots in source code

## Example Output

```
=== PROFILING RESULTS ===
Total samples: 3886

Function breakdown:
        evaluator.evaluate_hand         3815 (98.2%)
        main                           71   (1.8%)
```

This clearly identifies the bottleneck functions for optimization.