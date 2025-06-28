// Minimal Perfect Hash Function (CHD) implementation
// Pure primitives - no dependencies on specific table implementations

// Hash function used for CHD
inline fn mix64(x: u64, magic_constant: u64) u64 {
    const h = x *% magic_constant;
    return h ^ (h >> 29);
}

// CHD hash result type
pub const HashResult = struct {
    bucket: u32,
    base_index: u32,
};

// CHD lookup function (primitive)
pub inline fn lookup(key: u32, magic_constant: u64, g_array: []const u8, value_table: []const u16, table_size: u32) u16 {
    const h = mix64(@as(u64, key), magic_constant);
    const bucket = @as(u32, @intCast(h >> 51)); // Top 13 bits
    const base_index = @as(u32, @intCast(h & 0x1FFFF)); // Low 17 bits
    const displacement = g_array[bucket];
    const final_index = (base_index + displacement) & (table_size - 1);
    return value_table[final_index];
}

// CHD hash function for table construction
pub fn hash_key(key: u32, magic_constant: u64) HashResult {
    const h = mix64(@as(u64, key), magic_constant);
    return .{
        .bucket = @intCast(h >> 51),
        .base_index = @intCast(h & 0x1FFFF),
    };
}