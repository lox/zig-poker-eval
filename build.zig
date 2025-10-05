const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Work around Zig 0.15.1 self-hosted x86_64 backend bug with SIMD
    // The self-hosted backend can't encode vpmovzxbd with YMM registers
    // Use LLVM backend on x86_64 for proper SIMD support
    const use_llvm = if (target.result.cpu.arch == .x86_64) true else null;

    // Define core modules in dependency order

    // Level 1: Card module (no dependencies)
    const card_mod = b.addModule("card", .{
        .root_source_file = b.path("src/card.zig"),
    });

    // Level 2: Evaluator module (depends on card)
    const evaluator_mod = b.addModule("evaluator", .{
        .root_source_file = b.path("src/evaluator.zig"),
    });
    evaluator_mod.addImport("card", card_mod);

    // Level 3: Hand module (depends on card)
    const hand_mod = b.addModule("hand", .{
        .root_source_file = b.path("src/hand.zig"),
    });
    hand_mod.addImport("card", card_mod);

    // Level 3: Equity module (depends on card, evaluator)
    const equity_mod = b.addModule("equity", .{
        .root_source_file = b.path("src/equity.zig"),
    });
    equity_mod.addImport("card", card_mod);
    equity_mod.addImport("evaluator", evaluator_mod);

    // Level 4: Range module (depends on card, hand, equity)
    const range_mod = b.addModule("range", .{
        .root_source_file = b.path("src/range.zig"),
    });
    range_mod.addImport("card", card_mod);
    range_mod.addImport("hand", hand_mod);
    range_mod.addImport("equity", equity_mod);

    // Level 4: Analysis module (depends on card)
    const analysis_mod = b.addModule("analysis", .{
        .root_source_file = b.path("src/analysis.zig"),
    });
    analysis_mod.addImport("card", card_mod);

    // Level 4: Draws module (depends on card)
    const draws_mod = b.addModule("draws", .{
        .root_source_file = b.path("src/draws.zig"),
    });
    draws_mod.addImport("card", card_mod);

    // Level 5: Main poker module (depends on all others)
    const poker_mod = b.addModule("poker", .{
        .root_source_file = b.path("src/poker.zig"),
    });
    poker_mod.addImport("card", card_mod);
    poker_mod.addImport("evaluator", evaluator_mod);
    poker_mod.addImport("hand", hand_mod);
    poker_mod.addImport("range", range_mod);
    poker_mod.addImport("equity", equity_mod);
    poker_mod.addImport("analysis", analysis_mod);
    poker_mod.addImport("draws", draws_mod);

    // Tools module (depends on poker for benchmarking)
    const tools_mod = b.addModule("tools", .{
        .root_source_file = b.path("src/tools/benchmark.zig"),
    });
    tools_mod.addImport("poker", poker_mod);

    // Table builder executable (for manual table regeneration)
    const table_builder = b.addExecutable(.{
        .name = "build_tables",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/internal/build_tables.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add module path for slow evaluator (needed by table builder)
    const slow_evaluator = b.createModule(.{
        .root_source_file = b.path("src/internal/slow_evaluator.zig"),
    });
    slow_evaluator.addImport("card", card_mod);
    table_builder.root_module.addImport("slow_evaluator", slow_evaluator);
    table_builder.root_module.addImport("card", card_mod);

    // Run step to build lookup tables (manual use only)
    const build_tables = b.addRunArtifact(table_builder);
    const build_tables_step = b.step("build-tables", "Generate L1-optimized lookup tables");
    build_tables_step.dependOn(&build_tables.step);

    // Main CLI executable
    const exe = b.addExecutable(.{
        .name = "poker-eval",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .use_llvm = use_llvm,
    });
    // CLI only needs the main poker module (which provides everything)
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

    // Generate all hands tool
    const gen_all_hands = b.addExecutable(.{
        .name = "generate-all-hands",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/generate_all_hands.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    gen_all_hands.root_module.addImport("poker", poker_mod);
    b.installArtifact(gen_all_hands);

    const run_gen_all = b.addRunArtifact(gen_all_hands);
    const gen_all_step = b.step("gen-all", "Generate all 133M hand evaluations");
    gen_all_step.dependOn(&run_gen_all.step);

    // Verify all hands tool
    const verify_all_hands = b.addExecutable(.{
        .name = "verify-all-hands",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/verify_all_hands.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    verify_all_hands.root_module.addImport("poker", poker_mod);
    b.installArtifact(verify_all_hands);

    const run_verify_all = b.addRunArtifact(verify_all_hands);
    const verify_all_step = b.step("verify-all", "Verify evaluator against all 133M hands");
    verify_all_step.dependOn(&run_verify_all.step);

    // Test step - run tests from all modules
    const test_step = b.step("test", "Run all unit tests");

    // Test each module individually with proper dependencies

    // Card module tests (no dependencies)
    const card_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/card.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_card_tests = b.addRunArtifact(card_tests);
    test_step.dependOn(&run_card_tests.step);

    // Evaluator module tests (depends on card)
    const evaluator_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/evaluator.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .use_llvm = use_llvm,
    });
    evaluator_tests.root_module.addImport("card", card_mod);
    const run_evaluator_tests = b.addRunArtifact(evaluator_tests);
    test_step.dependOn(&run_evaluator_tests.step);

    // Hand module tests (depends on card)
    const hand_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/hand.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    hand_tests.root_module.addImport("card", card_mod);
    const run_hand_tests = b.addRunArtifact(hand_tests);
    test_step.dependOn(&run_hand_tests.step);

    // Range module tests (depends on card, hand, equity)
    const range_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/range.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .use_llvm = use_llvm,
    });
    range_tests.root_module.addImport("card", card_mod);
    range_tests.root_module.addImport("hand", hand_mod);
    range_tests.root_module.addImport("equity", equity_mod);
    const run_range_tests = b.addRunArtifact(range_tests);
    test_step.dependOn(&run_range_tests.step);

    // Equity module tests (depends on card, evaluator)
    const equity_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/equity.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .use_llvm = use_llvm,
    });
    equity_tests.root_module.addImport("card", card_mod);
    equity_tests.root_module.addImport("evaluator", evaluator_mod);
    const run_equity_tests = b.addRunArtifact(equity_tests);
    test_step.dependOn(&run_equity_tests.step);

    // Analysis module tests (depends on card)
    const analysis_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/analysis.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    analysis_tests.root_module.addImport("card", card_mod);
    const run_analysis_tests = b.addRunArtifact(analysis_tests);
    test_step.dependOn(&run_analysis_tests.step);

    // Draws module tests (depends on card)
    const draws_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/draws.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    draws_tests.root_module.addImport("card", card_mod);
    const run_draws_tests = b.addRunArtifact(draws_tests);
    test_step.dependOn(&run_draws_tests.step);

    // Main poker module tests (depends on all modules)
    const poker_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/poker.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .use_llvm = use_llvm,
    });
    poker_tests.root_module.addImport("card", card_mod);
    poker_tests.root_module.addImport("evaluator", evaluator_mod);
    poker_tests.root_module.addImport("hand", hand_mod);
    poker_tests.root_module.addImport("range", range_mod);
    poker_tests.root_module.addImport("equity", equity_mod);
    poker_tests.root_module.addImport("analysis", analysis_mod);
    poker_tests.root_module.addImport("draws", draws_mod);
    const run_poker_tests = b.addRunArtifact(poker_tests);
    test_step.dependOn(&run_poker_tests.step);
}
