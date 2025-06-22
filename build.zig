const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Table builder executable
    const table_builder = b.addExecutable(.{
        .name = "build_tables",
        .root_source_file = b.path("src/build_tables.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Run step to build lookup tables
    const build_tables = b.addRunArtifact(table_builder);
    const build_tables_step = b.step("build-tables", "Generate L1-optimized lookup tables");
    build_tables_step.dependOn(&build_tables.step);

    // Main evaluator executable
    const exe = b.addExecutable(.{
        .name = "poker-eval",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Ensure tables are built before main executable
    exe.step.dependOn(&build_tables.step);
    
    b.installArtifact(exe);

    // Benchmark executable
    const bench = b.addExecutable(.{
        .name = "bench",
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    bench.step.dependOn(&build_tables.step);
    
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

    // Test step
    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    exe_unit_tests.step.dependOn(&build_tables.step);

    // Evaluator test suite
    const evaluator_tests = b.addTest(.{
        .root_source_file = b.path("src/test_evaluator.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    evaluator_tests.step.dependOn(&build_tables.step);
    
    const run_evaluator_tests = b.addRunArtifact(evaluator_tests);
    const evaluator_test_step = b.step("test-evaluator", "Run comprehensive evaluator tests");
    evaluator_test_step.dependOn(&run_evaluator_tests.step);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
    test_step.dependOn(&run_evaluator_tests.step);
}
