# zig-poker-eval-mphf

This repository implements a high-performance, cache-efficient 7-card poker hand evaluator using minimal perfect hashing and SIMD techniques in Zig.

## Quick Start

To build and run the benchmarks (requires Zig 0.14.0 or later):

```sh
bin/zig build bench -Doptimize=ReleaseFast
```

To run tests:

```sh
bin/zig build test
```

## Assembly Analysis

To examine the generated assembly for performance analysis:

```sh
# Compile the evaluator with optimizations
zig build-exe src/real_eval_example.zig -O ReleaseFast -fno-strip --name evaluator_test

# Run the test to verify functionality
./evaluator_test

# View the assembly for the core evaluation function
objdump -d evaluator_test | sed -n '/evaluate_hand/,/^[0-9a-f]* <.*>:/p' | head -50
```

The assembly reveals:
- **NEON auto-vectorization** of rank computation using `cnt.8b` (popcount) and `ushl.2d` (vector shifts)
- **Optimal ARM64 arithmetic** with `x + (x << 2)` for multiply-by-5 in base-5 encoding
- **8-instruction critical path** for CHD lookup (multiply, shift, XOR, load displacement, add, mask, load rank)
- **Intelligent branching** between flush and non-flush evaluation paths

Key functions to analyze:
- `_evaluator.evaluate_hand` - Main evaluation with flush detection
- Critical path starts around the `mul x8, x8, x9` instruction (magic constant multiply)

## Documentation

- **[DESIGN.md](./DESIGN.md)** — Detailed design, algorithms, and implementation notes for the evaluator, including table generation, hashing, and SIMD strategies.
- **[BENCHMARKING.md](./BENCHMARKING.md)** — Methodology and validation procedures for benchmarking and correctness, including hardware setup, measurement, and validation techniques.
- **[EXPERIMENTS.md](./EXPERIMENTS.md)** — Performance optimization experiments and results, documenting what optimizations were tried and why they failed.

Refer to these documents for in-depth technical details, validation requirements, and reproducible benchmarking instructions.
