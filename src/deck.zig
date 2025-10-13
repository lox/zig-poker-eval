const std = @import("std");
const card = @import("card");

/// Full ordered deck of 52 unique cards encoded as single-bit masks.
pub const FULL_DECK = blk: {
    var deck: [52]card.Hand = undefined;
    for (0..52) |i| {
        deck[i] = @as(card.Hand, 1) << @intCast(i);
    }
    break :blk deck;
};

const INITIAL_POSITIONS = blk: {
    var pos: [52]u8 = undefined;
    for (0..52) |i| {
        pos[i] = @intCast(i);
    }
    break :blk pos;
};

const INVALID_POSITION: u8 = 0xFF;

/// Swap-remove sampler for drawing cards without replacement.
/// Copies the static FULL_DECK into an internal buffer so we can perform
/// O(1) draw operations that keep the remaining portion contiguous.
pub const DeckSampler = struct {
    cards: [52]card.Hand = FULL_DECK,
    positions: [52]u8 = INITIAL_POSITIONS,
    remaining: u8 = 52,

    /// Create a sampler seeded with the full deck.
    pub fn init() DeckSampler {
        return DeckSampler{
            .cards = FULL_DECK,
            .positions = INITIAL_POSITIONS,
            .remaining = 52,
        };
    }

    /// Create a sampler and immediately exclude `exclude_mask`.
    pub fn initWithMask(exclude_mask: card.Hand) DeckSampler {
        var sampler = DeckSampler.init();
        if (exclude_mask != 0) sampler.removeMask(exclude_mask);
        return sampler;
    }

    /// Reset to a full deck (no exclusions).
    pub fn reset(self: *DeckSampler) void {
        self.cards = FULL_DECK;
        self.positions = INITIAL_POSITIONS;
        self.remaining = 52;
    }

    /// Reset to a deck that excludes the provided mask of cards.
    pub fn resetWithMask(self: *DeckSampler, exclude_mask: card.Hand) void {
        self.reset();
        if (exclude_mask == 0) return;
        self.removeMask(exclude_mask);
    }

    /// Remove every card contained in `mask` (one bit per card).
    pub fn removeMask(self: *DeckSampler, mask: card.Hand) void {
        var remaining = mask;
        while (remaining != 0) {
            const bit_index: u6 = @intCast(@ctz(remaining));
            const card_mask = @as(card.Hand, 1) << bit_index;
            self.removeCard(card_mask);
            remaining &= remaining - 1;
        }
    }

    /// Remove a single specific card from the deck.
    pub fn removeCard(self: *DeckSampler, card_mask: card.Hand) void {
        std.debug.assert(card.countCards(card_mask) == 1);
        const card_index = cardIndex(card_mask);
        std.debug.assert(self.remaining > 0);
        const index = self.positions[card_index];
        std.debug.assert(index != INVALID_POSITION);

        const last_index = self.remaining - 1;
        if (index != last_index) {
            swapIndices(self, index, last_index);
        }

        self.remaining = last_index;
        self.positions[card_index] = INVALID_POSITION;
    }

    /// Draw a single card using swap-remove. `rng` must expose `uintLessThan`.
    pub fn draw(self: *DeckSampler, rng: anytype) card.Hand {
        std.debug.assert(self.remaining > 0);
        const idx = rng.uintLessThan(u8, self.remaining);
        const last_index = self.remaining - 1;
        if (idx != last_index) {
            swapIndices(self, idx, last_index);
        }

        const drawn = self.cards[last_index];
        self.positions[cardIndex(drawn)] = INVALID_POSITION;
        self.remaining = last_index;
        return drawn;
    }

    /// Draw multiple cards into `out` without replacement.
    pub fn drawMany(self: *DeckSampler, rng: anytype, out: []card.Hand) void {
        std.debug.assert(out.len <= self.remaining);
        for (out) |*slot| {
            slot.* = self.draw(rng);
        }
    }

    /// Draw `count` cards and return the combined bitmask.
    pub fn drawMask(self: *DeckSampler, rng: anytype, count: u8) card.Hand {
        if (count == 0) return 0;
        std.debug.assert(count <= self.remaining);

        var result: card.Hand = 0;
        var i: u8 = 0;
        while (i < count) : (i += 1) {
            result |= self.draw(rng);
        }
        return result;
    }

    /// Number of cards still available for drawing.
    pub fn remainingCards(self: DeckSampler) u8 {
        return self.remaining;
    }
};

inline fn cardIndex(card_mask: card.Hand) usize {
    std.debug.assert(card.countCards(card_mask) == 1);
    return @intCast(@ctz(card_mask));
}

fn swapIndices(self: *DeckSampler, idx_a: usize, idx_b: usize) void {
    if (idx_a == idx_b) return;

    const card_a = self.cards[idx_a];
    const card_b = self.cards[idx_b];
    self.cards[idx_a] = card_b;
    self.cards[idx_b] = card_a;

    self.positions[cardIndex(card_a)] = @intCast(idx_b);
    self.positions[cardIndex(card_b)] = @intCast(idx_a);
}

const testing = std.testing;

test "DeckSampler supports exclusions and unique draws" {
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rng = prng.random();

    var sampler = DeckSampler.init();
    // Remove two known cards.
    sampler.removeMask(card.makeCard(.spades, .ace) | card.makeCard(.hearts, .ace));
    try testing.expectEqual(@as(u8, 50), sampler.remainingCards());

    // Draw two distinct cards and ensure they are unique.
    const first = sampler.draw(rng);
    const second = sampler.draw(rng);
    try testing.expect((first & second) == 0);
    try testing.expectEqual(@as(u8, 48), sampler.remainingCards());
}

test "DeckSampler initWithMask excludes provided cards" {
    const exclude = card.makeCard(.spades, .ace) |
        card.makeCard(.hearts, .ace) |
        card.makeCard(.clubs, .king);

    var prng = std.Random.DefaultPrng.init(1);
    const rng = prng.random();

    var sampler = DeckSampler.initWithMask(exclude);
    try testing.expectEqual(@as(u8, 49), sampler.remainingCards());

    // Draw remaining cards and ensure excluded ones never appear.
    while (sampler.remainingCards() > 0) {
        const card_bit = sampler.draw(rng);
        try testing.expect((card_bit & exclude) == 0);
    }

    try testing.expectEqual(@as(u8, 0), sampler.remainingCards());
}
