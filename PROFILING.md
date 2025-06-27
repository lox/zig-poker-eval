# Profiling Scripts

Scripts for performance analysis and profiling of the poker hand evaluator.

## Scripts

### `simple_profile.sh`
**Primary profiling script** - High-frequency sampling with extended benchmark runs.

```bash
./scripts/simple_profile.sh
```

- Builds benchmark with ReleaseFast + debug symbols
- Runs 20M iterations (takes ~20 seconds)  
- Samples every 1ms for 15 seconds
- Outputs detailed function-level breakdown
- Generates `detailed_profile.txt`

**Use this for identifying performance bottlenecks.**

### `profile_bench.sh` 
Alternative profiling approach using zig build command directly.

```bash
./scripts/profile_bench.sh
```

- Uses `zig build bench` with custom iterations
- Samples for 20 seconds
- Good for quick profiling runs


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