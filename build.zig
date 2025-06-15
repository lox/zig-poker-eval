const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "poker",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Profiling executable
    const profile_exe = b.addExecutable(.{
        .name = "profile",
        .root_source_file = b.path("src/profile_main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(profile_exe);

    const profile_cmd = b.addRunArtifact(profile_exe);
    profile_cmd.step.dependOn(b.getInstallStep());

    const profile_step = b.step("profile", "Run performance profiling");
    profile_step.dependOn(&profile_cmd.step);

    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("src/bench_main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(benchmark_exe);

    const bench_cmd = b.addRunArtifact(benchmark_exe);
    bench_cmd.step.dependOn(b.getInstallStep());

    const bench_step = b.step("bench", "Run benchmarks");
    bench_step.dependOn(&bench_cmd.step);

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
