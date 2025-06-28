const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Define card module (source of truth for card format)
    const card_mod = b.addModule("card", .{
        .root_source_file = b.path("src/card/mod.zig"),
    });

    // Define evaluator module
    const evaluator_mod = b.addModule("evaluator", .{
        .root_source_file = b.path("src/evaluator/mod.zig"),
    });
    evaluator_mod.addImport("card", card_mod);

    // Define poker module
    const poker_mod = b.addModule("poker", .{
        .root_source_file = b.path("src/poker/mod.zig"),
    });
    poker_mod.addImport("card", card_mod);
    poker_mod.addImport("evaluator", evaluator_mod);

    // Define tools module
    const tools_mod = b.addModule("tools", .{
        .root_source_file = b.path("src/tools/benchmark.zig"),
    });
    tools_mod.addImport("evaluator", evaluator_mod);

    // Table builder executable (for manual table regeneration)
    const table_builder = b.addExecutable(.{
        .name = "build_tables",
        .root_source_file = b.path("src/evaluator/build_tables.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add module path for slow evaluator (needed by table builder)
    const slow_evaluator = b.createModule(.{
        .root_source_file = b.path("src/evaluator/slow_evaluator.zig"),
    });
    table_builder.root_module.addImport("slow_evaluator", slow_evaluator);

    // Run step to build lookup tables (manual use only)
    const build_tables = b.addRunArtifact(table_builder);
    const build_tables_step = b.step("build-tables", "Generate L1-optimized lookup tables");
    build_tables_step.dependOn(&build_tables.step);

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "poker-eval",
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("card", card_mod);
    exe.root_module.addImport("evaluator", evaluator_mod);
    exe.root_module.addImport("poker", poker_mod);
    exe.root_module.addImport("tools", tools_mod);

    b.installArtifact(exe);

    // Benchmark command (now via CLI)
    const run_bench_cmd = b.addRunArtifact(exe);
    run_bench_cmd.addArg("bench");
    if (b.args) |args| {
        run_bench_cmd.addArgs(args);
    }
    const bench_step = b.step("bench", "Run performance benchmark via CLI");
    bench_step.dependOn(&run_bench_cmd.step);

    // Run main executable
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the poker evaluator");
    run_step.dependOn(&run_cmd.step);

    // Test step - run tests from all modules
    const test_step = b.step("test", "Run all unit tests");

    // Test each module individually with proper dependencies

    // Card module tests (no dependencies)
    const card_tests = b.addTest(.{
        .root_source_file = b.path("src/card/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_card_tests = b.addRunArtifact(card_tests);
    test_step.dependOn(&run_card_tests.step);

    // Evaluator module tests (depends on card)
    const evaluator_tests = b.addTest(.{
        .root_source_file = b.path("src/evaluator/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    evaluator_tests.root_module.addImport("card", card_mod);
    const run_evaluator_tests = b.addRunArtifact(evaluator_tests);
    test_step.dependOn(&run_evaluator_tests.step);

    // Poker module tests (depends on card and evaluator)
    const poker_tests = b.addTest(.{
        .root_source_file = b.path("src/poker/mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    poker_tests.root_module.addImport("card", card_mod);
    poker_tests.root_module.addImport("evaluator", evaluator_mod);
    const run_poker_tests = b.addRunArtifact(poker_tests);
    test_step.dependOn(&run_poker_tests.step);
}
