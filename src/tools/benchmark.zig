const std = @import("std");
const builtin = @import("builtin");
const poker = @import("poker");

// ============================================================================
// Core Types
// ============================================================================

/// Statistics for a benchmark run
pub const Statistics = struct {
    mean: f64,
    median: f64,
    stddev: f64,
    cv: f64, // coefficient of variation
    min: f64,
    max: f64,
    runs: u32,

    pub fn isStable(self: Statistics) bool {
        return self.cv < 0.05; // CV < 5% = stable
    }
};

/// A single benchmark within a suite
pub const Benchmark = struct {
    name: []const u8,
    unit: []const u8,
    warmup_runs: u32 = 3,
    runs: u32 = 10,
    threshold_pct: ?f64 = null, // Per-benchmark regression threshold (e.g., 0.02 = 2%)
    run_fn: *const fn (allocator: std.mem.Allocator) anyerror!f64,
};

/// A suite of related benchmarks
pub const BenchmarkSuite = struct {
    name: []const u8,
    benchmarks: []const Benchmark,
};

/// Result of a single benchmark
pub const BenchmarkMetric = struct {
    unit: []const u8,
    value: f64,
    runs: u32,
    cv: f64,
};

/// System information for cross-platform comparison
pub const SystemInfo = struct {
    hostname: []const u8,
    cpu: []const u8,
    arch: []const u8,
    os: []const u8,
    build_mode: []const u8,

    pub fn deinit(self: *SystemInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.hostname);
        allocator.free(self.cpu);
        allocator.free(self.arch);
        allocator.free(self.os);
        allocator.free(self.build_mode);
    }
};

/// Complete baseline result
pub const Result = struct {
    version: []const u8 = "1.0",
    commit: ?[]const u8,
    timestamp: []const u8,
    system: SystemInfo,
    suites: std.StringHashMap(std.StringHashMap(BenchmarkMetric)),

    pub fn init(allocator: std.mem.Allocator) Result {
        return .{
            .commit = null,
            .timestamp = &.{},
            // SAFETY: system will be initialized by caller before use
            .system = undefined,
            .suites = std.StringHashMap(std.StringHashMap(BenchmarkMetric)).init(allocator),
        };
    }

    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        if (self.commit) |c| allocator.free(c);
        if (self.timestamp.len > 0) allocator.free(self.timestamp);
        self.system.deinit(allocator);

        var suite_iter = self.suites.iterator();
        while (suite_iter.next()) |entry| {
            var benchmark_map = entry.value_ptr.*;
            benchmark_map.deinit();
        }
        self.suites.deinit();
    }
};

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Runs a benchmark with warmup, multiple iterations, and statistics
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    runs_override: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, runs_override: ?u32) BenchmarkRunner {
        return .{ .allocator = allocator, .runs_override = runs_override };
    }

    pub fn run(self: BenchmarkRunner, benchmark: Benchmark) !Statistics {
        // Warmup runs
        for (0..benchmark.warmup_runs) |_| {
            _ = try benchmark.run_fn(self.allocator);
        }

        // Measurement runs (use override if provided)
        const num_runs = self.runs_override orelse benchmark.runs;
        const times = try self.allocator.alloc(f64, num_runs);
        defer self.allocator.free(times);

        for (times) |*time| {
            time.* = try benchmark.run_fn(self.allocator);
        }

        // Calculate statistics
        return try calculateStatistics(times);
    }
};

fn calculateStatistics(times: []f64) !Statistics {
    const n = times.len;
    if (n == 0) return error.EmptySample;

    // Sort in-place (we own the buffer)
    std.mem.sort(f64, times, {}, comptime std.sort.asc(f64));

    // Calculate IQM (Interquartile Mean): mean of middle 50% of samples
    // This eliminates outliers from OS scheduler interrupts and measurement noise
    const q1_idx = n / 4;
    const q3_idx = (3 * n) / 4;
    const iqm_samples = if (q3_idx > q1_idx) times[q1_idx..q3_idx] else times;

    var iqm_sum: f64 = 0;
    for (iqm_samples) |val| iqm_sum += val;
    const mean = iqm_sum / @as(f64, @floatFromInt(iqm_samples.len));

    // Calculate standard deviation from IQM samples only using sample variance (n-1)
    var variance: f64 = 0;
    for (iqm_samples) |time| {
        const diff = time - mean;
        variance += diff * diff;
    }
    const denom = if (iqm_samples.len > 1) iqm_samples.len - 1 else 1;
    variance /= @as(f64, @floatFromInt(denom));
    const stddev = @sqrt(variance);

    // Coefficient of variation from IQM
    const cv = if (mean != 0) stddev / mean else 0;

    // Median from sorted array (average of middle two for even length)
    const mid = n / 2;
    const median = if (n % 2 == 0)
        (times[mid - 1] + times[mid]) / 2.0
    else
        times[mid];

    // Min/max from sorted array
    const min_val = times[0];
    const max_val = times[n - 1];

    return .{
        .mean = mean,
        .median = median,
        .stddev = stddev,
        .cv = cv,
        .min = min_val,
        .max = max_val,
        .runs = @intCast(n),
    };
}

// ============================================================================
// System Info & Git Utilities
// ============================================================================

fn getSystemInfo(allocator: std.mem.Allocator) !SystemInfo {
    const hostname = try getHostname(allocator);
    const cpu = try getCpuModel(allocator);
    const arch = try allocator.dupe(u8, @tagName(builtin.cpu.arch));
    const os = try allocator.dupe(u8, @tagName(builtin.os.tag));
    const build_mode = try allocator.dupe(u8, @tagName(builtin.mode));

    return .{
        .hostname = hostname,
        .cpu = cpu,
        .arch = arch,
        .os = os,
        .build_mode = build_mode,
    };
}

fn getHostname(allocator: std.mem.Allocator) ![]const u8 {
    if (builtin.os.tag == .windows) {
        const env_val = std.process.getEnvVarOwned(allocator, "COMPUTERNAME") catch {
            return allocator.dupe(u8, "unknown");
        };
        return env_val;
    }

    var buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
    const hostname = std.posix.gethostname(&buf) catch {
        return allocator.dupe(u8, "unknown");
    };
    return allocator.dupe(u8, hostname);
}

fn getCpuModel(allocator: std.mem.Allocator) ![]const u8 {
    switch (builtin.os.tag) {
        .macos => {
            const result = std.process.Child.run(.{
                .allocator = allocator,
                .argv = &[_][]const u8{ "sysctl", "-n", "machdep.cpu.brand_string" },
            }) catch {
                return allocator.dupe(u8, "unknown");
            };
            defer allocator.free(result.stdout);
            defer allocator.free(result.stderr);

            if (result.term == .Exited and result.term.Exited == 0) {
                const trimmed = std.mem.trim(u8, result.stdout, &std.ascii.whitespace);
                return try allocator.dupe(u8, trimmed);
            }
            return allocator.dupe(u8, "unknown");
        },
        .linux => {
            const file = std.fs.cwd().openFile("/proc/cpuinfo", .{}) catch {
                return allocator.dupe(u8, "unknown");
            };
            defer file.close();

            const content = file.readToEndAlloc(allocator, 1 << 20) catch {
                return allocator.dupe(u8, "unknown");
            };
            defer allocator.free(content);

            var lines = std.mem.splitSequence(u8, content, "\n");
            while (lines.next()) |line| {
                if (std.mem.startsWith(u8, line, "model name")) {
                    if (std.mem.indexOf(u8, line, ":")) |colon_pos| {
                        const model = std.mem.trim(u8, line[colon_pos + 1 ..], &std.ascii.whitespace);
                        return try allocator.dupe(u8, model);
                    }
                }
            }
            return allocator.dupe(u8, "unknown");
        },
        else => return allocator.dupe(u8, "unknown"),
    }
}

fn getCurrentCommit(allocator: std.mem.Allocator) ?[]const u8 {
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &[_][]const u8{ "git", "describe", "--tags", "--dirty", "--always", "--abbrev=12", "--long" },
    }) catch return null;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.term == .Exited and result.term.Exited == 0) {
        const trimmed = std.mem.trim(u8, result.stdout, &std.ascii.whitespace);
        return allocator.dupe(u8, trimmed) catch null;
    }

    return null;
}

fn getTimestamp(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.timestamp();
    const seconds: u64 = @intCast(timestamp);

    // Format as ISO 8601: YYYY-MM-DDTHH:MM:SSZ
    const epoch_seconds = std.time.epoch.EpochSeconds{ .secs = seconds };
    const epoch_day = epoch_seconds.getEpochDay();
    const day_seconds = epoch_seconds.getDaySeconds();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    const hours = day_seconds.getHoursIntoDay();
    const minutes = day_seconds.getMinutesIntoHour();
    const secs = day_seconds.getSecondsIntoMinute();

    return std.fmt.allocPrint(allocator, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        year_day.year,
        month_day.month.numeric(),
        month_day.day_index + 1,
        hours,
        minutes,
        secs,
    });
}

/// Get the baseline file path for the current build mode
pub fn getBaselinePath(allocator: std.mem.Allocator) ![]const u8 {
    const mode_lower = switch (builtin.mode) {
        .Debug => "debug",
        .ReleaseSafe => "release-safe",
        .ReleaseFast => "release-fast",
        .ReleaseSmall => "release-small",
    };
    return std.fmt.allocPrint(allocator, "benchmark-baseline-{s}.json", .{mode_lower});
}

// ============================================================================
// Baseline Save/Load
// ============================================================================

pub fn saveBaseline(result: Result, path: []const u8) !void {
    // Build JSON string
    const allocator = result.suites.allocator;
    var string = try std.ArrayList(u8).initCapacity(allocator, 4096);
    defer string.deinit(allocator);

    const writer = string.writer(allocator);

    try string.appendSlice(allocator, "{\n");
    try std.fmt.format(writer, "  \"version\": \"{s}\",\n", .{result.version});

    if (result.commit) |commit| {
        try std.fmt.format(writer, "  \"commit\": \"{s}\",\n", .{commit});
    } else {
        try string.appendSlice(allocator, "  \"commit\": null,\n");
    }

    try std.fmt.format(writer, "  \"timestamp\": \"{s}\",\n", .{result.timestamp});

    try string.appendSlice(allocator, "  \"system\": {\n");
    try std.fmt.format(writer, "    \"hostname\": \"{s}\",\n", .{result.system.hostname});
    try std.fmt.format(writer, "    \"cpu\": \"{s}\",\n", .{result.system.cpu});
    try std.fmt.format(writer, "    \"arch\": \"{s}\",\n", .{result.system.arch});
    try std.fmt.format(writer, "    \"os\": \"{s}\",\n", .{result.system.os});
    try std.fmt.format(writer, "    \"build_mode\": \"{s}\"\n", .{result.system.build_mode});
    try string.appendSlice(allocator, "  },\n");

    try string.appendSlice(allocator, "  \"suites\": {\n");

    var suite_iter = result.suites.iterator();
    var first_suite = true;
    while (suite_iter.next()) |suite_entry| {
        if (!first_suite) try string.appendSlice(allocator, ",\n");
        first_suite = false;

        try std.fmt.format(writer, "    \"{s}\": {{\n", .{suite_entry.key_ptr.*});

        var bench_iter = suite_entry.value_ptr.iterator();
        var first_bench = true;
        while (bench_iter.next()) |bench_entry| {
            if (!first_bench) try string.appendSlice(allocator, ",\n");
            first_bench = false;

            const metric = bench_entry.value_ptr.*;
            try std.fmt.format(writer, "      \"{s}\": {{\n", .{bench_entry.key_ptr.*});
            try std.fmt.format(writer, "        \"unit\": \"{s}\",\n", .{metric.unit});
            try std.fmt.format(writer, "        \"value\": {d},\n", .{metric.value});
            try std.fmt.format(writer, "        \"runs\": {d},\n", .{metric.runs});
            try std.fmt.format(writer, "        \"cv\": {d}\n", .{metric.cv});
            try string.appendSlice(allocator, "      }");
        }
        try string.appendSlice(allocator, "\n    }");
    }

    try string.appendSlice(allocator, "\n  }\n}\n");

    // Write to file
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(string.items);
}

pub fn loadBaseline(path: []const u8, allocator: std.mem.Allocator) !Result {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(contents);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, contents, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    var result = Result.init(allocator);

    // Parse metadata
    if (root.get("commit")) |commit_val| {
        if (commit_val != .null) {
            result.commit = try allocator.dupe(u8, commit_val.string);
        }
    }

    result.timestamp = try allocator.dupe(u8, root.get("timestamp").?.string);

    // Parse system info
    const system_obj = root.get("system").?.object;
    result.system = .{
        .hostname = try allocator.dupe(u8, system_obj.get("hostname").?.string),
        .cpu = try allocator.dupe(u8, system_obj.get("cpu").?.string),
        .arch = try allocator.dupe(u8, system_obj.get("arch").?.string),
        .os = try allocator.dupe(u8, system_obj.get("os").?.string),
        .build_mode = try allocator.dupe(u8, system_obj.get("build_mode").?.string),
    };

    // Parse suites
    const suites_obj = root.get("suites").?.object;
    var suite_iter = suites_obj.iterator();
    while (suite_iter.next()) |suite_entry| {
        var benchmark_map = std.StringHashMap(BenchmarkMetric).init(allocator);

        const benchmarks_obj = suite_entry.value_ptr.object;
        var bench_iter = benchmarks_obj.iterator();
        while (bench_iter.next()) |bench_entry| {
            const metric_obj = bench_entry.value_ptr.object;
            const metric = BenchmarkMetric{
                .unit = try allocator.dupe(u8, metric_obj.get("unit").?.string),
                .value = metric_obj.get("value").?.float,
                .runs = @intCast(metric_obj.get("runs").?.integer),
                .cv = metric_obj.get("cv").?.float,
            };

            try benchmark_map.put(try allocator.dupe(u8, bench_entry.key_ptr.*), metric);
        }

        try result.suites.put(try allocator.dupe(u8, suite_entry.key_ptr.*), benchmark_map);
    }

    return result;
}

// ============================================================================
// Comparison Engine
// ============================================================================

pub const ComparisonResult = struct {
    passed: bool,
    regressions: std.ArrayList(Regression),
    improvements: std.ArrayList(Improvement),
    missing_benchmarks: std.ArrayList(MissingBenchmark),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ComparisonResult) void {
        self.regressions.deinit(self.allocator);
        self.improvements.deinit(self.allocator);
        self.missing_benchmarks.deinit(self.allocator);
    }
};

pub const Regression = struct {
    suite: []const u8,
    benchmark: []const u8,
    baseline_value: f64,
    current_value: f64,
    change_pct: f64,
    unit: []const u8,
};

pub const Improvement = struct {
    suite: []const u8,
    benchmark: []const u8,
    baseline_value: f64,
    current_value: f64,
    change_pct: f64,
    unit: []const u8,
};

pub const MissingBenchmark = struct {
    suite: []const u8,
    benchmark: []const u8,
};

pub fn compareResults(baseline: Result, current: Result, default_threshold_pct: f64, allocator: std.mem.Allocator) !ComparisonResult {
    var result = ComparisonResult{
        .passed = true,
        .regressions = try std.ArrayList(Regression).initCapacity(allocator, 0),
        .improvements = try std.ArrayList(Improvement).initCapacity(allocator, 0),
        .missing_benchmarks = try std.ArrayList(MissingBenchmark).initCapacity(allocator, 0),
        .allocator = allocator,
    };

    // Check for build mode mismatch (hard error)
    if (!std.mem.eql(u8, baseline.system.build_mode, current.system.build_mode)) {
        std.debug.print("❌ ERROR: Build mode mismatch!\n", .{});
        std.debug.print("   Baseline: {s}\n", .{baseline.system.build_mode});
        std.debug.print("   Current:  {s}\n", .{current.system.build_mode});
        std.debug.print("\n", .{});
        std.debug.print("   Cannot compare benchmarks across different build modes.\n", .{});
        std.debug.print("   Each build mode has its own baseline file.\n", .{});
        return error.BuildModeMismatch;
    }

    // Warn if system mismatch
    if (!std.mem.eql(u8, baseline.system.hostname, current.system.hostname) or
        !std.mem.eql(u8, baseline.system.cpu, current.system.cpu))
    {
        std.debug.print("⚠️  Warning: Baseline from different system\n", .{});
        std.debug.print("   Baseline: {s} ({s})\n", .{ baseline.system.hostname, baseline.system.cpu });
        std.debug.print("   Current:  {s} ({s})\n", .{ current.system.hostname, current.system.cpu });
        std.debug.print("\n", .{});
    }

    // Compare each suite and benchmark
    var suite_iter = current.suites.iterator();
    while (suite_iter.next()) |suite_entry| {
        const suite_name = suite_entry.key_ptr.*;
        const current_benchmarks = suite_entry.value_ptr.*;

        // Check if baseline has this suite
        const baseline_benchmarks = baseline.suites.get(suite_name) orelse continue;

        var bench_iter = current_benchmarks.iterator();
        while (bench_iter.next()) |bench_entry| {
            const bench_name = bench_entry.key_ptr.*;
            const current_metric = bench_entry.value_ptr.*;

            // Check if baseline has this benchmark
            const baseline_metric = baseline_benchmarks.get(bench_name) orelse continue;

            // Look up per-benchmark threshold from suite definitions
            var effective_threshold = default_threshold_pct;
            for (ALL_SUITES) |suite| {
                if (std.mem.eql(u8, suite.name, suite_name)) {
                    for (suite.benchmarks) |benchmark| {
                        if (std.mem.eql(u8, benchmark.name, bench_name)) {
                            effective_threshold = benchmark.threshold_pct orelse default_threshold_pct;
                            break;
                        }
                    }
                    break;
                }
            }

            // Apply stability guard: if CV >= 5%, use max(threshold, 2*CV) to avoid false positives
            if (current_metric.cv >= 0.05) {
                effective_threshold = @max(effective_threshold, 2.0 * current_metric.cv);
            }

            // Calculate percentage change (as decimal, not percentage)
            const change_decimal = (current_metric.value - baseline_metric.value) / baseline_metric.value;

            // Regression is positive change (slower/worse)
            if (change_decimal > effective_threshold) {
                result.passed = false;
                try result.regressions.append(allocator, .{
                    .suite = suite_name,
                    .benchmark = bench_name,
                    .baseline_value = baseline_metric.value,
                    .current_value = current_metric.value,
                    .change_pct = change_decimal * 100.0,
                    .unit = current_metric.unit,
                });
            } else if (change_decimal < -effective_threshold) {
                // Improvement is negative change (faster/better)
                try result.improvements.append(allocator, .{
                    .suite = suite_name,
                    .benchmark = bench_name,
                    .baseline_value = baseline_metric.value,
                    .current_value = current_metric.value,
                    .change_pct = change_decimal * 100.0,
                    .unit = current_metric.unit,
                });
            }
        }
    }

    // Check for benchmarks in baseline that are missing from current results
    var baseline_suite_iter = baseline.suites.iterator();
    while (baseline_suite_iter.next()) |baseline_suite_entry| {
        const baseline_suite_name = baseline_suite_entry.key_ptr.*;
        const baseline_benchmarks = baseline_suite_entry.value_ptr.*;

        // Check if current has this suite
        const current_benchmarks = current.suites.get(baseline_suite_name);
        if (current_benchmarks == null) {
            // Entire suite is missing - add all benchmarks from that suite
            var baseline_bench_iter = baseline_benchmarks.iterator();
            while (baseline_bench_iter.next()) |baseline_bench_entry| {
                try result.missing_benchmarks.append(allocator, .{
                    .suite = baseline_suite_name,
                    .benchmark = baseline_bench_entry.key_ptr.*,
                });
            }
            continue;
        }

        // Suite exists, check individual benchmarks
        var baseline_bench_iter = baseline_benchmarks.iterator();
        while (baseline_bench_iter.next()) |baseline_bench_entry| {
            const baseline_bench_name = baseline_bench_entry.key_ptr.*;

            if (current_benchmarks.?.get(baseline_bench_name) == null) {
                try result.missing_benchmarks.append(allocator, .{
                    .suite = baseline_suite_name,
                    .benchmark = baseline_bench_name,
                });
            }
        }
    }

    return result;
}

pub fn printComparisonResult(comparison: ComparisonResult) void {
    std.debug.print("📊 Comparison vs Baseline\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});

    if (comparison.improvements.items.len > 0) {
        std.debug.print("\n✅ Improvements:\n", .{});
        for (comparison.improvements.items) |imp| {
            std.debug.print("  {s}/{s}: {d:.2} → {d:.2} {s} ({d:.1}%)\n", .{
                imp.suite,
                imp.benchmark,
                imp.baseline_value,
                imp.current_value,
                imp.unit,
                imp.change_pct,
            });
        }
    }

    if (comparison.regressions.items.len > 0) {
        std.debug.print("\n❌ Regressions:\n", .{});
        for (comparison.regressions.items) |reg| {
            std.debug.print("  {s}/{s}: {d:.2} → {d:.2} {s} (+{d:.1}%)\n", .{
                reg.suite,
                reg.benchmark,
                reg.baseline_value,
                reg.current_value,
                reg.unit,
                reg.change_pct,
            });
        }
    }

    if (comparison.regressions.items.len == 0 and comparison.improvements.items.len == 0) {
        std.debug.print("\n✓ No significant changes\n", .{});
    }

    std.debug.print("\n", .{});
    if (comparison.passed) {
        std.debug.print("✅ PASSED\n", .{});
    } else {
        std.debug.print("❌ FAILED - {d} regression(s) detected\n", .{comparison.regressions.items.len});
    }
}

// ============================================================================
// Benchmark Helpers
// ============================================================================

const BATCH_SIZE = 32;
const MULTIWAY_SEATS = 6;

const ShowdownCase = struct {
    ctx: poker.BoardContext,
    hero_hole: u64,
    villain_hole: u64,
};

const MultiwayCase = struct {
    ctx: poker.BoardContext,
    seats: [MULTIWAY_SEATS]poker.Hand,
};

fn createBatch(hands: []const u64, start_idx: usize) @Vector(BATCH_SIZE, u64) {
    var batch_hands: [BATCH_SIZE]u64 = undefined;
    for (0..BATCH_SIZE) |i| {
        batch_hands[i] = hands[(start_idx + i) % hands.len];
    }
    const batch: @Vector(BATCH_SIZE, u64) = batch_hands;
    return batch;
}

fn drawUniqueCard(rng: std.Random, used: *u64) u64 {
    while (true) {
        const idx = rng.uintLessThan(u6, 52);
        const card_bit: u64 = @as(u64, 1) << @intCast(idx);
        if ((used.* & card_bit) == 0) {
            used.* |= card_bit;
            return card_bit;
        }
    }
}

fn generateShowdownCases(allocator: std.mem.Allocator, iterations: u32, rng: std.Random) ![]ShowdownCase {
    const cases = try allocator.alloc(ShowdownCase, iterations);
    errdefer allocator.free(cases);

    var index: usize = 0;
    while (index < cases.len) {
        const remaining = cases.len - index;
        const group_size = @min(BATCH_SIZE, remaining);

        var board_used: u64 = 0;
        var board: u64 = 0;
        while (@popCount(board) < 5) {
            board |= drawUniqueCard(rng, &board_used);
        }
        const ctx = poker.initBoardContext(board);

        var pair: usize = 0;
        while (pair < group_size) : (pair += 1) {
            var used: u64 = board;

            var hero_hole: u64 = 0;
            while (@popCount(hero_hole) < 2) {
                hero_hole |= drawUniqueCard(rng, &used);
            }

            var villain_hole: u64 = 0;
            while (@popCount(villain_hole) < 2) {
                villain_hole |= drawUniqueCard(rng, &used);
            }

            cases[index + pair] = .{
                .ctx = ctx,
                .hero_hole = hero_hole,
                .villain_hole = villain_hole,
            };
        }

        index += group_size;
    }

    return cases;
}

fn timeScalarShowdown(cases: []const ShowdownCase) f64 {
    var checksum: i32 = 0;
    var timer = std.time.Timer.start() catch unreachable;
    for (cases) |case| {
        checksum += poker.evaluateShowdownWithContext(&case.ctx, case.hero_hole, case.villain_hole);
    }
    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);
    const cases_len: f64 = @floatFromInt(cases.len);
    return total_ns / cases_len;
}

fn generateMultiwayCases(allocator: std.mem.Allocator, iterations: u32, rng: std.Random) ![]MultiwayCase {
    const cases = try allocator.alloc(MultiwayCase, iterations);
    errdefer allocator.free(cases);

    for (cases) |*case| {
        var used: u64 = 0;

        var board: u64 = 0;
        while (@popCount(board) < 5) {
            board |= drawUniqueCard(rng, &used);
        }

        var seats: [MULTIWAY_SEATS]poker.Hand = undefined;
        inline for (0..MULTIWAY_SEATS) |seat_idx| {
            var hole: u64 = 0;
            while (@popCount(hole) < 2) {
                hole |= drawUniqueCard(rng, &used);
            }
            seats[seat_idx] = hole;
        }

        case.* = .{
            .ctx = poker.initBoardContext(board),
            .seats = seats,
        };
    }

    return cases;
}

fn benchMultiwayShowdown(allocator: std.mem.Allocator) !f64 {
    const iterations = 100000;
    const repeats = 10;

    var prng = std.Random.DefaultPrng.init(314);
    const rng = prng.random();

    const cases = try generateMultiwayCases(allocator, iterations, rng);
    defer allocator.free(cases);

    var timer = try std.time.Timer.start();
    var checksum: u64 = 0;

    for (0..repeats) |_| {
        for (cases) |case| {
            const result = poker.evaluateShowdownMultiway(&case.ctx, &case.seats);
            checksum +%= result.best_rank;
            checksum ^= result.winner_mask;
        }
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const total_ops: f64 = @floatFromInt(@as(u64, iterations) * repeats);
    return total_ns / total_ops;
}

fn benchMultiwayEquityWeights(allocator: std.mem.Allocator) !f64 {
    const iterations = 100000;
    const repeats = 10;

    var prng = std.Random.DefaultPrng.init(2718);
    const rng = prng.random();

    const cases = try generateMultiwayCases(allocator, iterations, rng);
    defer allocator.free(cases);

    var timer = try std.time.Timer.start();
    var checksum: f64 = 0.0;
    var equities: [MULTIWAY_SEATS]f64 = undefined;

    for (0..repeats) |_| {
        for (cases) |case| {
            const result = poker.evaluateEquityWeights(&case.ctx, &case.seats, equities[0..]);
            checksum += @as(f64, @floatFromInt(result.tie_count));
            for (equities) |eq| checksum += eq;
        }
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const total_ops: f64 = @floatFromInt(@as(u64, iterations) * repeats);
    return total_ns / total_ops;
}

fn timeBatchedShowdown(cases: []const ShowdownCase) f64 {
    var checksum: i32 = 0;
    var timer = std.time.Timer.start() catch unreachable;
    var index: usize = 0;
    // SAFETY: buffers initialized before use in loop below
    var results_buffer: [BATCH_SIZE]i8 = undefined;
    var hero_buffer: [BATCH_SIZE]poker.Hand = undefined;
    var villain_buffer: [BATCH_SIZE]poker.Hand = undefined;

    while (index < cases.len) {
        const remaining = cases.len - index;
        const chunk = @min(BATCH_SIZE, remaining);

        inline for (0..BATCH_SIZE) |i| {
            if (i < chunk) {
                const case_ref = cases[index + i];
                std.debug.assert(case_ref.ctx.board == cases[index].ctx.board);
                hero_buffer[i] = case_ref.hero_hole;
                villain_buffer[i] = case_ref.villain_hole;
            }
        }

        poker.evaluateShowdownBatch(&cases[index].ctx, hero_buffer[0..chunk], villain_buffer[0..chunk], results_buffer[0..chunk]);
        for (results_buffer[0..chunk]) |res| {
            checksum += res;
        }

        index += chunk;
    }
    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);
    const cases_len: f64 = @floatFromInt(cases.len);
    return total_ns / cases_len;
}

// ============================================================================
// Benchmark Suite Definitions
// ============================================================================

fn benchEvalBatch(allocator: std.mem.Allocator) !f64 {
    const iterations = 100000;
    const num_test_hands = 100000;

    // Generate test hands
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    const test_hands = try allocator.alloc(u64, num_test_hands);
    defer allocator.free(test_hands);

    for (test_hands) |*hand| {
        hand.* = poker.generateRandomHand(&rng);
    }

    // Benchmark
    var checksum: u64 = 0;
    var hand_idx: usize = 0;
    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        const batch = createBatch(test_hands, hand_idx);
        const results = poker.evaluateBatch(BATCH_SIZE, batch);
        for (0..BATCH_SIZE) |j| {
            checksum +%= results[j];
        }
        hand_idx = (hand_idx + BATCH_SIZE) % num_test_hands;
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const total_hands: f64 = @floatFromInt(iterations * BATCH_SIZE);
    return total_ns / total_hands;
}

fn benchShowdownContext(allocator: std.mem.Allocator) !f64 {
    const iterations = 100000;
    const repeats = 10; // Repeat measurement 10x for stability

    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const cases = try generateShowdownCases(allocator, iterations, rng);
    defer allocator.free(cases);

    var timer = try std.time.Timer.start();

    for (0..repeats) |_| {
        const result = timeScalarShowdown(cases);
        std.mem.doNotOptimizeAway(result);
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    const total_ops: f64 = @floatFromInt(iterations * repeats);
    return total_ns / total_ops;
}

fn benchShowdownBatch(allocator: std.mem.Allocator) !f64 {
    const iterations = 100000;
    const repeats = 10; // Repeat measurement 10x for stability

    var prng = std.Random.DefaultPrng.init(99);
    const rng = prng.random();

    const cases = try generateShowdownCases(allocator, iterations, rng);
    defer allocator.free(cases);

    var timer = try std.time.Timer.start();

    for (0..repeats) |_| {
        const result = timeBatchedShowdown(cases);
        std.mem.doNotOptimizeAway(result);
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    const total_ops: f64 = @floatFromInt(iterations * repeats);
    return total_ns / total_ops;
}

fn benchEquityMonteCarlo(allocator: std.mem.Allocator) !f64 {
    const iterations = 1000;
    const simulations = 10000;

    // Fixed seed for reproducibility (but RNG state advances across iterations)
    var prng = std.Random.DefaultPrng.init(123);
    const rng = prng.random();

    // AA vs KK - classic matchup
    const aa = poker.makeCard(.clubs, .ace) | poker.makeCard(.diamonds, .ace);
    const kk = poker.makeCard(.hearts, .king) | poker.makeCard(.spades, .king);

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        const result = try poker.monteCarlo(aa, kk, &.{}, simulations, rng, allocator);
        std.mem.doNotOptimizeAway(result);
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    const iterations_f64: f64 = @floatFromInt(iterations);
    const ns_per_calc = total_ns / iterations_f64;
    return ns_per_calc / 1000.0; // Convert to microseconds
}

fn benchEquityExact(allocator: std.mem.Allocator) !f64 {
    const iterations = 200;
    const repeats = 100; // Repeat 100x for stability (deterministic, needs low variance)

    // AA vs KK on turn (only 44 rivers to enumerate)
    const aa = poker.makeCard(.clubs, .ace) | poker.makeCard(.diamonds, .ace);
    const kk = poker.makeCard(.hearts, .king) | poker.makeCard(.spades, .king);
    const board = [_]poker.Hand{
        poker.makeCard(.spades, .seven),
        poker.makeCard(.hearts, .eight),
        poker.makeCard(.diamonds, .nine),
        poker.makeCard(.clubs, .two),
    };

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        for (0..repeats) |_| {
            const result = try poker.exact(aa, kk, &board, allocator);
            std.mem.doNotOptimizeAway(result);
        }
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    const total_ops: f64 = @floatFromInt(iterations * repeats);
    const ns_per_calc = total_ns / total_ops;
    return ns_per_calc / 1000.0; // Convert to microseconds
}

fn benchRangeEquityMonteCarlo(allocator: std.mem.Allocator) !f64 {
    const iterations = 10;
    const repeats = 10; // Repeat 10x for stability
    const simulations = 1000;

    // Fixed seed for reproducibility (but RNG state advances across iterations)
    var prng = std.Random.DefaultPrng.init(456);
    const rng = prng.random();

    // Create ranges once
    var hero_range = poker.Range.init(allocator);
    defer hero_range.deinit();
    try hero_range.addHandNotation("AA", 1.0);

    var villain_range = poker.Range.init(allocator);
    defer villain_range.deinit();
    try villain_range.addHandNotation("KK", 1.0);

    var timer = try std.time.Timer.start();

    for (0..iterations) |_| {
        for (0..repeats) |_| {
            const result = try hero_range.equityMonteCarlo(&villain_range, &.{}, simulations, rng, allocator);
            std.mem.doNotOptimizeAway(result);
        }
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    const total_ops: f64 = @floatFromInt(iterations * repeats);
    const ns_per_calc = total_ns / total_ops;
    return ns_per_calc / 1_000_000.0; // Convert to milliseconds
}

// ============================================================================
// Microbenchmarks (Hot Path Profiling)
// ============================================================================

fn benchInitBoardContext(allocator: std.mem.Allocator) !f64 {
    const iterations = 1_000_000;
    const num_boards = 10000;

    // Generate test boards (3-5 cards each)
    var prng = std.Random.DefaultPrng.init(77);
    var rng = prng.random();
    const boards = try allocator.alloc(u64, num_boards);
    defer allocator.free(boards);

    for (boards) |*board_slot| {
        const num_cards = 3 + (rng.uintLessThan(u8, 3)); // 3, 4, or 5 cards
        var board_used: u64 = 0;
        var board: u64 = 0;
        while (@popCount(board) < num_cards) {
            board |= drawUniqueCard(rng, &board_used);
        }
        board_slot.* = board;
    }

    // Benchmark
    var checksum: u64 = 0;
    var timer = try std.time.Timer.start();

    for (0..iterations) |i| {
        const board = boards[i % num_boards];
        const ctx = poker.initBoardContext(board);
        checksum +%= ctx.suit_counts[0];
        checksum +%= ctx.rank_counts[0];
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const iterations_f64: f64 = @floatFromInt(iterations);
    return total_ns / iterations_f64;
}

fn benchEvaluateHoleWithContext(allocator: std.mem.Allocator) !f64 {
    const iterations = 1_000_000;
    const num_cases = 10000;

    // Generate test cases: board contexts + hole cards
    var prng = std.Random.DefaultPrng.init(88);
    const rng = prng.random();

    const HoleCase = struct {
        ctx: poker.BoardContext,
        hole: u64,
    };

    const cases = try allocator.alloc(HoleCase, num_cases);
    defer allocator.free(cases);

    for (cases) |*case| {
        var board_used: u64 = 0;
        var board: u64 = 0;
        while (@popCount(board) < 5) {
            board |= drawUniqueCard(rng, &board_used);
        }
        const ctx = poker.initBoardContext(board);

        var hole: u64 = 0;
        while (@popCount(hole) < 2) {
            hole |= drawUniqueCard(rng, &board_used);
        }

        case.* = .{ .ctx = ctx, .hole = hole };
    }

    // Benchmark
    var checksum: u64 = 0;
    var timer = try std.time.Timer.start();

    for (0..iterations) |i| {
        const case = cases[i % num_cases];
        const rank = poker.evaluateHoleWithContext(&case.ctx, case.hole);
        checksum +%= rank;
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const iterations_f64: f64 = @floatFromInt(iterations);
    return total_ns / iterations_f64;
}

fn benchEvaluateHandSingle(allocator: std.mem.Allocator) !f64 {
    const iterations = 1_000_000;
    const num_test_hands = 10000;

    // Generate test hands
    var prng = std.Random.DefaultPrng.init(66);
    var rng = prng.random();
    const test_hands = try allocator.alloc(u64, num_test_hands);
    defer allocator.free(test_hands);

    for (test_hands) |*hand| {
        hand.* = poker.generateRandomHand(&rng);
    }

    // Benchmark
    var checksum: u64 = 0;
    var timer = try std.time.Timer.start();

    for (0..iterations) |i| {
        const hand = test_hands[i % num_test_hands];
        const rank = poker.evaluateHand(hand);
        checksum +%= rank;
    }

    const total_ns: f64 = @floatFromInt(timer.read());
    std.mem.doNotOptimizeAway(checksum);

    const iterations_f64: f64 = @floatFromInt(iterations);
    return total_ns / iterations_f64;
}

pub const ALL_SUITES = [_]BenchmarkSuite{
    .{
        .name = "eval",
        .benchmarks = &.{
            .{
                .name = "batch_evaluation",
                .unit = "ns/hand",
                .threshold_pct = 0.02, // 2% - tight threshold for core SIMD evaluator
                .run_fn = benchEvalBatch,
            },
            .{
                .name = "single_evaluation",
                .unit = "ns/hand",
                .threshold_pct = 0.03, // 3% - scalar path
                .run_fn = benchEvaluateHandSingle,
            },
        },
    },
    .{
        .name = "context",
        .benchmarks = &.{
            .{
                .name = "init_board",
                .unit = "ns/call",
                .threshold_pct = 0.03, // 3% - deterministic operation
                .run_fn = benchInitBoardContext,
            },
            .{
                .name = "hole_evaluation",
                .unit = "ns/eval",
                .threshold_pct = 0.03, // 3% - deterministic
                .run_fn = benchEvaluateHoleWithContext,
            },
        },
    },
    .{
        .name = "showdown",
        .benchmarks = &.{
            .{
                .name = "context_path",
                .unit = "ns/eval",
                .threshold_pct = 0.025, // 2.5% - deterministic
                .run_fn = benchShowdownContext,
            },
            .{
                .name = "batched",
                .unit = "ns/eval",
                .threshold_pct = 0.025, // 2.5% - deterministic
                .run_fn = benchShowdownBatch,
            },
        },
    },
    .{
        .name = "multiway",
        .benchmarks = &.{
            .{
                .name = "showdown_multiway",
                .unit = "ns/eval",
                .threshold_pct = 0.03,
                .run_fn = benchMultiwayShowdown,
            },
            .{
                .name = "equity_weights",
                .unit = "ns/eval",
                .threshold_pct = 0.03,
                .run_fn = benchMultiwayEquityWeights,
            },
        },
    },
    .{
        .name = "equity",
        .benchmarks = &.{
            .{
                .name = "monte_carlo",
                .unit = "µs/calc",
                .threshold_pct = 0.08, // 8% - Monte Carlo has natural randomness
                .run_fn = benchEquityMonteCarlo,
            },
            .{
                .name = "exact_turn",
                .unit = "µs/calc",
                .warmup_runs = 5,
                .runs = 20, // More runs for stability
                .threshold_pct = 0.03, // 3% - deterministic enumeration
                .run_fn = benchEquityExact,
            },
        },
    },
    .{
        .name = "range",
        .benchmarks = &.{
            .{
                .name = "equity_monte_carlo",
                .unit = "ms/calc",
                .threshold_pct = 0.12, // 12% - highest variance from range × MC
                .run_fn = benchRangeEquityMonteCarlo,
            },
        },
    },
};

// ============================================================================
// Throughput Formatting
// ============================================================================

fn formatThroughput(buffer: *[32]u8, value: f64, unit: []const u8) []const u8 {
    // Convert to operations per second based on unit
    if (std.mem.endsWith(u8, unit, "ns/hand") or std.mem.endsWith(u8, unit, "ns/eval")) {
        // nanoseconds per operation → ops/sec = 1e9 / ns
        const ops_per_sec = 1_000_000_000.0 / value;
        if (ops_per_sec >= 1_000_000_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}G/s", .{ops_per_sec / 1_000_000_000.0}) catch "?";
        } else if (ops_per_sec >= 1_000_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}M/s", .{ops_per_sec / 1_000_000.0}) catch "?";
        } else if (ops_per_sec >= 1_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}K/s", .{ops_per_sec / 1_000.0}) catch "?";
        } else {
            return std.fmt.bufPrint(buffer, "{d:.2}/s", .{ops_per_sec}) catch "?";
        }
    } else if (std.mem.endsWith(u8, unit, "µs/calc") or std.mem.endsWith(u8, unit, "us/calc")) {
        // microseconds per operation → ops/sec = 1e6 / µs
        const ops_per_sec = 1_000_000.0 / value;
        if (ops_per_sec >= 1_000_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}M/s", .{ops_per_sec / 1_000_000.0}) catch "?";
        } else if (ops_per_sec >= 1_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}K/s", .{ops_per_sec / 1_000.0}) catch "?";
        } else {
            return std.fmt.bufPrint(buffer, "{d:.2}/s", .{ops_per_sec}) catch "?";
        }
    } else if (std.mem.endsWith(u8, unit, "ms/calc")) {
        // milliseconds per operation → ops/sec = 1e3 / ms
        const ops_per_sec = 1_000.0 / value;
        if (ops_per_sec >= 1_000.0) {
            return std.fmt.bufPrint(buffer, "{d:.2}K/s", .{ops_per_sec / 1_000.0}) catch "?";
        } else {
            return std.fmt.bufPrint(buffer, "{d:.2}/s", .{ops_per_sec}) catch "?";
        }
    } else {
        // Unknown unit - show raw value with unit to avoid misleading throughput
        return std.fmt.bufPrint(buffer, "{d:.2} {s}", .{ value, unit }) catch "?";
    }
}

// ============================================================================
// Public API
// ============================================================================

pub fn runAllBenchmarks(allocator: std.mem.Allocator, filter: ?[]const u8, for_baseline: bool) !Result {
    var result = Result.init(allocator);
    result.system = try getSystemInfo(allocator);
    result.commit = getCurrentCommit(allocator);
    result.timestamp = try getTimestamp(allocator);

    // Use 100 runs for baseline, 10 for normal benchmarks
    const runs_override: ?u32 = if (for_baseline) 100 else null;
    const runner = BenchmarkRunner.init(allocator, runs_override);

    // Header
    std.debug.print("\n", .{});
    std.debug.print("🚀 Running Benchmarks\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("Build mode: {s}\n", .{result.system.build_mode});
    if (result.commit) |commit| {
        std.debug.print("Version:    {s}\n", .{commit});
    }
    if (for_baseline) {
        std.debug.print("Mode:       Baseline (100 runs per benchmark)\n", .{});
    }
    std.debug.print("\n", .{});

    for (ALL_SUITES, 0..) |suite, suite_idx| {
        // Apply filter if specified
        if (filter) |f| {
            if (!std.mem.eql(u8, suite.name, f)) continue;
        }

        std.debug.print("[{d}/{d}] {s}\n", .{ suite_idx + 1, ALL_SUITES.len, suite.name });

        var benchmark_map = std.StringHashMap(BenchmarkMetric).init(allocator);
        var suite_timer = try std.time.Timer.start();

        for (suite.benchmarks) |benchmark| {
            const stats = try runner.run(benchmark);

            // Show result with value
            std.debug.print("  • {s}: {d:.2} {s}", .{ benchmark.name, stats.median, benchmark.unit });

            // Calculate and show throughput (ops/second)
            var throughput_buf: [32]u8 = undefined;
            const throughput = formatThroughput(&throughput_buf, stats.median, benchmark.unit);
            std.debug.print(" ({s})", .{throughput});

            if (!stats.isStable()) {
                std.debug.print(" ⚠️  CV={d:.1}%", .{stats.cv * 100.0});
            }
            std.debug.print("\n", .{});

            try benchmark_map.put(benchmark.name, .{
                .unit = benchmark.unit,
                .value = stats.median,
                .runs = stats.runs,
                .cv = stats.cv,
            });
        }

        // Show suite timing
        const suite_time_ns = suite_timer.read();
        const suite_time_s = @as(f64, @floatFromInt(suite_time_ns)) / 1_000_000_000.0;
        if (suite_time_s < 10.0) {
            std.debug.print("  ↳ Ran in {d:.1}s\n", .{suite_time_s});
        } else {
            std.debug.print("  ↳ Ran in {d:.0}s\n", .{suite_time_s});
        }

        try result.suites.put(suite.name, benchmark_map);
        std.debug.print("\n", .{}); // Blank line between suites
    }

    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("\n", .{});
    return result;
}
