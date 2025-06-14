const std = @import("std");
const poker = @import("poker.zig");

// Generate random 7-card hands for fair benchmarking
pub fn generateRandomHands(allocator: std.mem.Allocator, count: u32, seed: u64) ![]poker.Hand {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    
    const hands = try allocator.alloc(poker.Hand, count);
    
    for (hands) |*hand| {
        hand.* = generateRandomHand(random);
    }
    
    return hands;
}

fn generateRandomHand(random: std.Random) poker.Hand {
    var hand = poker.Hand.init();
    var used_cards = std.StaticBitSet(52).initEmpty();
    
    // Generate 7 unique random cards
    var cards_added: u8 = 0;
    while (cards_added < 7) {
        const card_idx = random.uintLessThan(u8, 52);
        if (!used_cards.isSet(card_idx)) {
            used_cards.set(card_idx);
            
            // Convert card index to rank/suit
            const rank: u8 = (card_idx / 4) + 2; // 0-51 -> ranks 2-14
            const suit: u2 = @intCast(card_idx % 4);
            
            hand.addCard(poker.Card.init(rank, suit));
            cards_added += 1;
        }
    }
    
    return hand;
}

// Generate torture case hands (edge cases) using clean card notation
pub fn generateTortureCases(allocator: std.mem.Allocator) ![]poker.Hand {
    const hands = try allocator.alloc(poker.Hand, 11);
    
    // Using runtime parseCards for clean, readable card definitions  
    hands[0] = poker.parseCards("AsKsQsJsTs2h3d") catch unreachable;  // Royal flush + 2 random
    hands[1] = poker.parseCards("9h8h7h6h5h2s3d") catch unreachable;  // Straight flush + 2 random  
    hands[2] = poker.parseCards("AhAsAdAc2h3s4d") catch unreachable;  // Four of a kind + 3 random
    hands[3] = poker.parseCards("AhAsAdKhKs2c3d") catch unreachable;  // Full house + 2 random
    hands[4] = poker.parseCards("AhQhTh8h6h2s3d") catch unreachable;  // Flush + 2 random
    hands[5] = poker.parseCards("AhKsQdJcTh2s3d") catch unreachable;  // Straight + 2 random
    hands[6] = poker.parseCards("AhAsAdKhQs2c3d") catch unreachable;  // Three of a kind + 4 random
    hands[7] = poker.parseCards("AhAsKhKsQh2c3d") catch unreachable;  // Two pair + 3 random
    hands[8] = poker.parseCards("AhAsKhQsJd2c3d") catch unreachable;  // One pair + 5 random
    hands[9] = poker.parseCards("AhKsQdJc9h7s2d") catch unreachable;  // High card + 6 random
    hands[10] = poker.parseCards("Ah5s4d3c2h6s7d") catch unreachable; // Wheel straight (A-2-3-4-5) + 2 random
    
    return hands;
}
