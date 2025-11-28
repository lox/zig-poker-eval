const std = @import("std");
const card = @import("card");
const features_mod = @import("features");
const analysis = @import("analysis");

pub const HandFeatures = features_mod.HandFeatures;
pub const HISTOGRAM_BINS = features_mod.HISTOGRAM_BINS;

/// Distance metric for comparing hands
pub const DistanceMetric = enum {
    /// Fast: based on draw types and strength difference
    feature_based,

    /// Slower but more accurate: EMD on equity histograms
    earth_movers,

    /// Hybrid: feature-based with EMD tiebreaker
    hybrid,
};

/// Poker street for table generation
pub const Street = enum(u2) {
    preflop = 0,
    flop = 1,
    turn = 2,
    river = 3,
};

/// Compute distance between two hands using the specified metric.
/// All distance functions are allocation-free.
pub fn handDistance(
    a: HandFeatures,
    b: HandFeatures,
    metric: DistanceMetric,
) f32 {
    return switch (metric) {
        .feature_based => featureDistance(a, b),
        .earth_movers => blk: {
            if (!a.has_equity_histogram or !b.has_equity_histogram) {
                break :blk featureDistance(a, b);
            }
            break :blk earthMoversDistance(&a.equity_histogram, &b.equity_histogram);
        },
        .hybrid => blk: {
            const feat_dist = featureDistance(a, b);
            if (!a.has_equity_histogram or !b.has_equity_histogram) {
                break :blk feat_dist;
            }
            const emd = earthMoversDistance(&a.equity_histogram, &b.equity_histogram);
            break :blk feat_dist * 0.7 + emd * 0.3;
        },
    };
}

/// Feature-based distance (fast, no histogram required)
fn featureDistance(a: HandFeatures, b: HandFeatures) f32 {
    var dist: f32 = 0.0;

    // Strength difference (scaled to be primary component)
    dist += @abs(a.strength - b.strength) * 2.0;

    // Made hand category mismatch
    if (a.made_category != b.made_category) {
        dist += 0.5;
    }

    // Draw type mismatches (binary features)
    if ((a.has_flush_draw or a.has_nut_flush_draw) != (b.has_flush_draw or b.has_nut_flush_draw)) {
        dist += 0.5;
    }
    if (a.has_nut_flush_draw != b.has_nut_flush_draw) {
        dist += 0.3;
    }
    if (a.has_oesd != b.has_oesd) {
        dist += 0.4;
    }
    if (a.has_gutshot != b.has_gutshot) {
        dist += 0.2;
    }
    if (a.has_backdoor_flush != b.has_backdoor_flush) {
        dist += 0.1;
    }
    if (a.has_backdoor_straight != b.has_backdoor_straight) {
        dist += 0.1;
    }

    // Outs difference
    const outs_diff: i16 = @as(i16, a.outs) - @as(i16, b.outs);
    dist += @as(f32, @floatFromInt(if (outs_diff < 0) -outs_diff else outs_diff)) * 0.1;

    // Board texture mismatch
    if (a.board_texture != b.board_texture) {
        dist += 0.2;
    }

    return dist;
}

/// Earth Mover's Distance for 1D histograms (O(n))
/// Works on any equal-length histograms, used internally and exposed for custom use.
pub fn earthMoversDistance(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var cumulative: f32 = 0.0;
    var total: f32 = 0.0;

    for (a, b) |ai, bi| {
        cumulative += ai - bi;
        total += @abs(cumulative);
    }

    return total;
}

/// Configuration for the bucketer
pub const BucketerConfig = struct {
    k: u32, // number of buckets
    metric: DistanceMetric = .feature_based,
    max_iterations: u32 = 100,
    seed: u64 = 0,

    // For equity-based bucketing
    use_equity_histogram: bool = false,
    simulations_per_card: u32 = 100,
};

/// Sample with optional weight for CFR reach-weighting
const WeightedSample = struct {
    features: HandFeatures,
    weight: f32,
};

/// K-means bucketer for hand abstraction
pub const Bucketer = struct {
    config: BucketerConfig,
    centroids: []HandFeatures,
    samples: std.ArrayList(WeightedSample),
    allocator: std.mem.Allocator,
    is_fitted: bool,

    pub fn init(config: BucketerConfig, allocator: std.mem.Allocator) Bucketer {
        return Bucketer{
            .config = config,
            .centroids = &.{},
            .samples = .empty,
            .allocator = allocator,
            .is_fitted = false,
        };
    }

    pub fn deinit(self: *Bucketer) void {
        if (self.centroids.len > 0) {
            self.allocator.free(self.centroids);
        }
        self.samples.deinit(self.allocator);
    }

    /// Add training samples with optional weight (for CFR reach-weighting).
    /// weight=null means uniform weight (1.0).
    pub fn addSample(self: *Bucketer, feat: HandFeatures, weight: ?f32) !void {
        try self.samples.append(self.allocator, .{
            .features = feat,
            .weight = weight orelse 1.0,
        });
    }

    /// Run k-means clustering. Call after adding all samples.
    pub fn fit(self: *Bucketer) !void {
        const n = self.samples.items.len;
        const k = self.config.k;

        if (n == 0) return error.NoSamples;
        if (k == 0) return error.ZeroBuckets;
        if (k > n) return error.MoreBucketsThanSamples;

        // Free old centroids if re-fitting
        if (self.centroids.len > 0) {
            self.allocator.free(self.centroids);
        }

        // Allocate centroids
        self.centroids = try self.allocator.alloc(HandFeatures, k);
        errdefer self.allocator.free(self.centroids);

        // Initialize centroids using k-means++ strategy
        try self.initCentroidsKMeansPlusPlus();

        // Allocate assignment array
        const assignments = try self.allocator.alloc(u32, n);
        defer self.allocator.free(assignments);

        // First assignment pass: no comparison, just fill
        for (self.samples.items, 0..) |sample, i| {
            assignments[i] = self.findNearestCentroid(sample.features);
        }
        try self.updateCentroids(assignments);

        // Run k-means iterations
        var iteration: u32 = 1;
        var changed = true;

        while (changed and iteration < self.config.max_iterations) {
            changed = false;

            // Assignment step: assign each sample to nearest centroid
            for (self.samples.items, 0..) |sample, i| {
                const new_assignment = self.findNearestCentroid(sample.features);
                if (assignments[i] != new_assignment) {
                    assignments[i] = new_assignment;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update step: recalculate centroids as weighted mean
            try self.updateCentroids(assignments);

            iteration += 1;
        }

        self.is_fitted = true;
    }

    /// Initialize centroids using k-means++ for better convergence
    fn initCentroidsKMeansPlusPlus(self: *Bucketer) !void {
        const n = self.samples.items.len;
        const k = self.config.k;

        var prng = std.Random.DefaultPrng.init(self.config.seed);
        const rng = prng.random();

        // Allocate distance array
        var distances = try self.allocator.alloc(f32, n);
        defer self.allocator.free(distances);

        // First centroid: random sample
        const first_idx = rng.uintLessThan(usize, n);
        self.centroids[0] = self.samples.items[first_idx].features;

        // Remaining centroids: probability proportional to distance squared
        for (1..k) |c| {
            var total_dist: f32 = 0.0;

            // Calculate distances to nearest centroid
            for (self.samples.items, 0..) |sample, i| {
                var min_dist: f32 = std.math.floatMax(f32);
                for (0..c) |prev_c| {
                    const d = handDistance(sample.features, self.centroids[prev_c], self.config.metric);
                    min_dist = @min(min_dist, d);
                }
                distances[i] = min_dist * min_dist; // Square for probability weighting
                total_dist += distances[i];
            }

            // Sample next centroid proportional to distance
            if (total_dist > 0) {
                const threshold = rng.float(f32) * total_dist;
                var selected_idx: usize = 0;
                var cumulative: f32 = 0.0;

                for (distances, 0..) |d, i| {
                    cumulative += d;
                    if (cumulative >= threshold) {
                        selected_idx = i;
                        break;
                    }
                }

                self.centroids[c] = self.samples.items[selected_idx].features;
            } else {
                // All points at same location, pick random
                self.centroids[c] = self.samples.items[rng.uintLessThan(usize, n)].features;
            }
        }
    }

    /// Update centroids as weighted mean of assigned samples
    fn updateCentroids(self: *Bucketer, assignments: []const u32) !void {
        const k = self.config.k;

        // Allocate weight accumulators
        var weights = try self.allocator.alloc(f32, k);
        defer self.allocator.free(weights);
        @memset(weights, 0.0);

        // Accumulators for each feature (simplified: just use strength as proxy)
        var strength_sums = try self.allocator.alloc(f32, k);
        defer self.allocator.free(strength_sums);
        @memset(strength_sums, 0.0);

        // Accumulate weighted sums
        for (self.samples.items, 0..) |sample, i| {
            const cluster = assignments[i];
            weights[cluster] += sample.weight;
            strength_sums[cluster] += sample.features.strength * sample.weight;
        }

        // Update centroids (preserving structure, updating key features)
        for (0..k) |c| {
            if (weights[c] > 0) {
                // For simplicity, update strength; in practice would update all features
                // The centroid inherits structure from first assigned sample
                var found_sample = false;
                for (self.samples.items, 0..) |sample, i| {
                    if (assignments[i] == c and !found_sample) {
                        self.centroids[c] = sample.features;
                        found_sample = true;
                    }
                }
                self.centroids[c].strength = strength_sums[c] / weights[c];
            }
        }
    }

    /// Find the index of the nearest centroid
    fn findNearestCentroid(self: *const Bucketer, feat: HandFeatures) u32 {
        var min_dist: f32 = std.math.floatMax(f32);
        var nearest: u32 = 0;

        for (self.centroids, 0..) |centroid, i| {
            const d = handDistance(feat, centroid, self.config.metric);
            if (d < min_dist) {
                min_dist = d;
                nearest = @intCast(i);
            }
        }

        return nearest;
    }

    /// Assign a hand to a bucket (0 to k-1). No allocation.
    pub fn assign(self: *const Bucketer, feat: HandFeatures) u32 {
        std.debug.assert(self.is_fitted);
        return self.findNearestCentroid(feat);
    }

    /// Batch assign for efficiency. No allocation.
    pub fn assignBatch(
        self: *const Bucketer,
        feats: []const HandFeatures,
        out: []u32,
    ) void {
        std.debug.assert(self.is_fitted);
        std.debug.assert(feats.len == out.len);

        for (feats, 0..) |feat, i| {
            out[i] = self.findNearestCentroid(feat);
        }
    }
};

/// Binary file header - extern struct for mmap compatibility.
/// Explicit padding ensures stable layout across compilers.
pub const TableHeader = extern struct {
    magic: [4]u8 = .{ 'B', 'U', 'C', 'K' },
    version: u32 = 1,
    street: u8,
    _pad: [3]u8 = .{ 0, 0, 0 }, // explicit padding (offset 9)
    k: u32,
    entry_count: u64,
    checksum: u64,
    _reserved: [32]u8 = .{0} ** 32, // (offset 32) - total 64 bytes, cache-line aligned
};

/// Pre-computed bucket lookup table
pub const Table = struct {
    header: TableHeader,
    buckets: []const u32, // mmap'd or loaded
    allocator: ?std.mem.Allocator,
    owned: bool, // true if we own the bucket memory

    /// Load table from file
    pub fn load(path: []const u8, allocator: std.mem.Allocator) !Table {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read header
        var header: TableHeader = undefined;
        const header_bytes = std.mem.asBytes(&header);
        const bytes_read = try file.readAll(header_bytes);
        if (bytes_read != @sizeOf(TableHeader)) {
            return error.InvalidTableFile;
        }

        // Validate magic
        if (!std.mem.eql(u8, &header.magic, "BUCK")) {
            return error.InvalidMagic;
        }

        // Read bucket data
        const entry_count: usize = @intCast(header.entry_count);
        const buckets = try allocator.alloc(u32, entry_count);
        errdefer allocator.free(buckets);

        const bucket_bytes = std.mem.sliceAsBytes(buckets);
        const bucket_read = try file.readAll(bucket_bytes);
        if (bucket_read != bucket_bytes.len) {
            return error.TruncatedTableFile;
        }

        return Table{
            .header = header,
            .buckets = buckets,
            .allocator = allocator,
            .owned = true,
        };
    }

    /// Save table to file
    pub fn save(self: *const Table, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Write header
        const header_bytes = std.mem.asBytes(&self.header);
        try file.writeAll(header_bytes);

        // Write bucket data
        const bucket_bytes = std.mem.sliceAsBytes(self.buckets);
        try file.writeAll(bucket_bytes);
    }

    pub fn deinit(self: *Table) void {
        if (self.owned) {
            if (self.allocator) |alloc| {
                alloc.free(self.buckets);
            }
        }
    }

    /// Quick lookup for preflop hands. No allocation.
    /// @param hole_hand Combined 2-card hole hand bitmask
    pub fn getBucket(self: *const Table, hole_hand: card.Hand) u32 {
        const idx = indexOfHand(hole_hand);
        if (idx >= self.buckets.len) return 0;
        return self.buckets[idx];
    }

    /// Lookup for postflop. No allocation.
    /// This is a simplified implementation - production would use board abstraction.
    pub fn getBucketPostflop(
        self: *const Table,
        hole_hand: card.Hand,
        board: []const card.Hand,
    ) u32 {
        _ = board; // Board abstraction not yet implemented
        return self.getBucket(hole_hand);
    }

    /// Integer index for a 2-card hole hand (0-168 canonical index)
    /// Maps to the 169 unique starting hands using the standard grid:
    /// - Diagonal: pocket pairs (AA=168, KK=155, ..., 22=0)
    /// - Upper triangle: suited hands (AKs=167, AQs=166, ...)
    /// - Lower triangle: offsuit hands (AKo=155, AQo=142, ...)
    /// Asserts that exactly 2 cards are in the hand.
    pub fn indexOfHand(hole_hand: card.Hand) u16 {
        std.debug.assert(card.countCards(hole_hand) == 2);

        // Extract the two cards
        var remaining = hole_hand;
        const bit1: u6 = @intCast(@ctz(remaining));
        remaining &= remaining - 1;
        const bit2: u6 = @intCast(@ctz(remaining));

        // Get ranks and suits
        const rank1: u8 = bit1 % 13;
        const suit1: u8 = bit1 / 13;
        const rank2: u8 = bit2 % 13;
        const suit2: u8 = bit2 / 13;

        // Canonical ordering: higher rank first
        const high_rank = @max(rank1, rank2);
        const low_rank = @min(rank1, rank2);
        const suited = (suit1 == suit2);

        if (high_rank == low_rank) {
            // Pocket pair - on the diagonal
            return @as(u16, high_rank) * 13 + high_rank;
        } else if (suited) {
            // Suited hand - upper triangle (row > col)
            return @as(u16, high_rank) * 13 + low_rank;
        } else {
            // Offsuit hand - lower triangle (col > row)
            return @as(u16, low_rank) * 13 + high_rank;
        }
    }

    /// Board abstraction index (placeholder - not yet implemented)
    pub fn indexOfBoard(board: []const card.Hand) u32 {
        _ = board;
        return 0; // Board abstraction is a separate module
    }
};

/// Builder for generating bucket tables offline
pub const TableBuilder = struct {
    street: Street,
    config: BucketerConfig,
    allocator: std.mem.Allocator,

    pub fn init(street: Street, config: BucketerConfig, allocator: std.mem.Allocator) TableBuilder {
        return TableBuilder{
            .street = street,
            .config = config,
            .allocator = allocator,
        };
    }

    /// Generate table by sampling hands and clustering.
    /// For flop: samples N random flops, clusters hands on each.
    pub fn build(
        self: *TableBuilder,
        board_samples: u32,
        rng: std.Random,
    ) !Table {
        _ = board_samples; // TODO: implement board sampling
        _ = rng;

        // For preflop, we just use the 169 canonical hands
        if (self.street == .preflop) {
            return self.buildPreflop();
        }

        // TODO: implement postflop table building
        return error.NotImplemented;
    }

    /// Build preflop table (169 entries, one per canonical hand)
    fn buildPreflop(self: *TableBuilder) !Table {
        const k = self.config.k;

        // For preflop, if k >= 169, each hand gets its own bucket
        if (k >= 169) {
            const buckets = try self.allocator.alloc(u32, 169);
            for (buckets, 0..) |*b, i| {
                b.* = @intCast(i);
            }

            return Table{
                .header = TableHeader{
                    .street = @intFromEnum(self.street),
                    .k = @intCast(@min(k, 169)),
                    .entry_count = 169,
                    .checksum = 0, // TODO: compute checksum
                },
                .buckets = buckets,
                .allocator = self.allocator,
                .owned = true,
            };
        }

        // TODO: implement k-means clustering for preflop
        return error.NotImplemented;
    }
};

// Tests
const testing = std.testing;

test "feature-based distance" {
    const f1 = HandFeatures.extract(
        card.parseCard("As") | card.parseCard("Ks"),
        &[_]card.Hand{
            card.parseCard("Ah"),
            card.parseCard("7c"),
            card.parseCard("2d"),
        },
    );
    const f2 = HandFeatures.extract(
        card.parseCard("Ad") | card.parseCard("Kh"),
        &[_]card.Hand{
            card.parseCard("Ah"),
            card.parseCard("7c"),
            card.parseCard("2d"),
        },
    );

    // Same hand class should have zero distance
    const dist = handDistance(f1, f2, .feature_based);
    try testing.expect(dist < 0.1);

    // Different hands should have larger distance
    const f3 = HandFeatures.extract(
        card.parseCard("2h") | card.parseCard("7s"),
        &[_]card.Hand{
            card.parseCard("Ah"),
            card.parseCard("Kc"),
            card.parseCard("Qd"),
        },
    );

    const dist2 = handDistance(f1, f3, .feature_based);
    try testing.expect(dist2 > dist);
}

test "earth movers distance" {
    // Distribution shifted by 1 bin (both sum to 1.0)
    const a = [_]f32{ 0.1, 0.2, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 0.1, 0.2, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const emd = earthMoversDistance(&a, &b);
    try testing.expect(emd > 0);
    // EMD for shift of 1 bin should be moderate (each unit of mass travels 1 bin)
    try testing.expect(emd < 2.0);

    // Same distribution should have zero EMD
    const emd_same = earthMoversDistance(&a, &a);
    try testing.expect(emd_same < 0.001);
}

test "bucketer basic flow" {
    var bucketer = Bucketer.init(.{
        .k = 3,
        .metric = .feature_based,
        .max_iterations = 10,
    }, testing.allocator);
    defer bucketer.deinit();

    // Add some samples
    const boards = [_][3]card.Hand{
        .{ card.parseCard("Ah"), card.parseCard("7c"), card.parseCard("2d") },
        .{ card.parseCard("Kh"), card.parseCard("Qc"), card.parseCard("Jd") },
    };

    const hands = [_]card.Hand{
        card.parseCard("As") | card.parseCard("Ks"),
        card.parseCard("Ad") | card.parseCard("Kh"),
        card.parseCard("Qd") | card.parseCard("Jh"),
        card.parseCard("2s") | card.parseCard("3s"),
        card.parseCard("7h") | card.parseCard("8h"),
        card.parseCard("Ts") | card.parseCard("9s"),
    };

    for (hands) |h| {
        for (boards) |b| {
            const feat = HandFeatures.extract(h, &b);
            try bucketer.addSample(feat, null);
        }
    }

    // Fit the bucketer
    try bucketer.fit();
    try testing.expect(bucketer.is_fitted);

    // Assign should work
    const feat = HandFeatures.extract(hands[0], &boards[0]);
    const bucket = bucketer.assign(feat);
    try testing.expect(bucket < 3);
}

test "table header size" {
    // Ensure header is exactly 64 bytes (cache-line aligned)
    try testing.expectEqual(@as(usize, 64), @sizeOf(TableHeader));
}

test "indexOfHand canonical mapping" {
    // Pocket aces (AA) - should be on diagonal
    const aa = card.parseCard("As") | card.parseCard("Ah");
    const aa_idx = Table.indexOfHand(aa);
    try testing.expectEqual(@as(u16, 12 * 13 + 12), aa_idx); // A=12, so 12*13+12=168

    // AKs (suited) - upper triangle
    const aks = card.parseCard("As") | card.parseCard("Ks");
    const aks_idx = Table.indexOfHand(aks);
    try testing.expectEqual(@as(u16, 12 * 13 + 11), aks_idx); // A=12, K=11, suited: 12*13+11=167

    // AKo (offsuit) - lower triangle
    const ako = card.parseCard("As") | card.parseCard("Kd");
    const ako_idx = Table.indexOfHand(ako);
    try testing.expectEqual(@as(u16, 11 * 13 + 12), ako_idx); // low*13+high = 11*13+12=155
}

test "table save and load" {
    // Create a simple table
    const buckets = try testing.allocator.alloc(u32, 169);
    defer testing.allocator.free(buckets);

    for (buckets, 0..) |*b, i| {
        b.* = @intCast(i % 10);
    }

    var table = Table{
        .header = TableHeader{
            .street = 0,
            .k = 10,
            .entry_count = 169,
            .checksum = 0,
        },
        .buckets = buckets,
        .allocator = null,
        .owned = false,
    };

    // Save to temp file
    const tmp_path = "/tmp/test_bucket_table.bin";
    try table.save(tmp_path);

    // Load it back
    var loaded = try Table.load(tmp_path, testing.allocator);
    defer loaded.deinit();

    try testing.expectEqual(@as(u64, 169), loaded.header.entry_count);
    try testing.expectEqual(@as(u32, 10), loaded.header.k);

    // Verify bucket values
    for (loaded.buckets, 0..) |b, i| {
        try testing.expectEqual(@as(u32, @intCast(i % 10)), b);
    }
}

test {
    std.testing.refAllDecls(@This());
}
