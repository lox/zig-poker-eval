# zig-poker-eval-mphf

This repository implements a high-performance, cache-efficient 7-card poker hand evaluator using minimal perfect hashing and SIMD techniques in Zig.

## Quick Start

To build and run the benchmarks (requires Zig 0.14.0 or later):

```sh
bin/zig build bench -Doptimize=ReleaseFast -Dcpu=native
```

See [BENCHMARKING.md](./BENCHMARKING.md) for detailed benchmarking methodology and validation steps.

## Documentation

- **[DESIGN.md](./DESIGN.md)** — Detailed design, algorithms, and implementation notes for the evaluator, including table generation, hashing, and SIMD strategies.
- **[BENCHMARKING.md](./BENCHMARKING.md)** — Methodology and validation procedures for benchmarking and correctness, including hardware setup, measurement, and validation techniques.

Refer to these documents for in-depth technical details, validation requirements, and reproducible benchmarking instructions.
