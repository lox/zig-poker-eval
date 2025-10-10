// Placeholder heads-up equity matrix (169x169)
// Each entry is (win_rate_x1000, tie_rate_x1000)
//
// TODO: Run `task gen:heads-up-matrix` to generate real data (~20 min)
//
// This placeholder uses the vs-random approximation for now

const heads_up_tables = @import("heads_up_tables.zig");

pub const HEADS_UP_MATRIX: [169][169][2]u16 = blk: {
    @setEvalBranchQuota(100000); // Need higher quota for 169x169 iterations
    var matrix: [169][169][2]u16 = undefined;

    // Initialize with placeholder using vs-random approximation
    for (0..169) |i| {
        for (0..169) |j| {
            const hero_eq = heads_up_tables.PREFLOP_VS_RANDOM[i];
            const villain_eq = heads_up_tables.PREFLOP_VS_RANDOM[j];

            // Rough approximation until real data is generated
            if (i == j) {
                // Same hand class - approximately 50/50
                matrix[i][j] = .{ 500, hero_eq[1] };
            } else {
                // Different hands - approximate based on vs-random
                const hero_win: u32 = hero_eq[0];
                const villain_win: u32 = villain_eq[0];
                const total = hero_win + villain_win;

                if (total > 0) {
                    const norm_win = (hero_win * 1000) / total;
                    const avg_tie = (hero_eq[1] + villain_eq[1]) / 2;
                    matrix[i][j] = .{ @intCast(norm_win), @intCast(avg_tie) };
                } else {
                    matrix[i][j] = .{ 500, 0 };
                }
            }
        }
    }

    break :blk matrix;
};
