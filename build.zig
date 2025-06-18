const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zig-poker-eval",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Legacy bench command - now use: zig-poker-eval bench
    const bench_cmd = b.addRunArtifact(exe);
    bench_cmd.step.dependOn(b.getInstallStep());
    bench_cmd.addArgs(&.{"bench"});

    if (b.args) |args| {
        bench_cmd.addArgs(args);
    }

    const bench_step = b.step("bench", "Run benchmarks via CLI");
    bench_step.dependOn(&bench_cmd.step);

    // MPHF generator for build-time table creation
    const mphf_generator_exe = b.addExecutable(.{
        .name = "mphf-generator",
        .root_source_file = b.path("build_tools/standalone_mphf_generator.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mphf_generator_cmd = b.addRunArtifact(mphf_generator_exe);
    const mphf_generator_step = b.step("generate-tables", "Generate MPHF lookup tables");
    mphf_generator_step.dependOn(&mphf_generator_cmd.step);

    // Tests
    const test_step = b.step("test", "Run unit tests");
    for ([_][]const u8{
        "src/poker.zig",
        "src/simulation.zig",
        "src/equity.zig",
        "src/ranges.zig",
        "src/benchmark.zig",
    }) |test_file| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(test_file),
            .target = target,
            .optimize = optimize,
        });
        test_step.dependOn(&b.addRunArtifact(unit_tests).step);
    }
}
