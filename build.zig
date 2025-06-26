const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Table builder executable (for manual table regeneration)
    const table_builder = b.addExecutable(.{
        .name = "build_tables",
        .root_source_file = b.path("src/build_tables.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add module path for slow evaluator (needed by table builder)
    const slow_evaluator = b.createModule(.{
        .root_source_file = b.path("src/slow_evaluator.zig"),
    });
    table_builder.root_module.addImport("slow_evaluator", slow_evaluator);

    // Run step to build lookup tables (manual use only)
    const build_tables = b.addRunArtifact(table_builder);
    const build_tables_step = b.step("build-tables", "Generate L1-optimized lookup tables");
    build_tables_step.dependOn(&build_tables.step);

    // Main evaluator executable (uses pre-compiled tables)
    const exe = b.addExecutable(.{
        .name = "poker-eval",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // NO LONGER DEPENDS ON TABLE BUILDING - uses pre-compiled tables.zig
    b.installArtifact(exe);

    // Benchmark executable (uses pre-compiled tables)
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
    });

    // NO LONGER DEPENDS ON TABLE BUILDING - uses pre-compiled tables.zig

    // Run benchmark
    const run_bench = b.addRunArtifact(bench);
    const bench_step = b.step("bench", "Run performance benchmark");
    bench_step.dependOn(&run_bench.step);

    // Run main executable
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the poker evaluator");
    run_step.dependOn(&run_cmd.step);

    // Test step (uses pre-compiled tables)
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // SIMD evaluator tests
    const simd_tests = b.addTest(.{
        .root_source_file = b.path("src/simd_evaluator.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Validation tests
    const validation_tests = b.addTest(.{
        .root_source_file = b.path("src/validation.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    const run_simd_tests = b.addRunArtifact(simd_tests);
    const run_validation_tests = b.addRunArtifact(validation_tests);

    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
    test_step.dependOn(&run_simd_tests.step);
    test_step.dependOn(&run_validation_tests.step);
}
