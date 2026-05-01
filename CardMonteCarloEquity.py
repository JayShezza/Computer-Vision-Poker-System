from itertools import combinations
from collections import Counter
import time
import random

CARD_VALUE = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}

FULL_DECK = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
    'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS'
]

HIGH_CARD      = 0
PAIR           = 1
TWO_PAIR       = 2
THREE_OF_KIND  = 3
STRAIGHT       = 4
FLUSH          = 5
FULL_HOUSE     = 6
FOUR_OF_KIND   = 7
STRAIGHT_FLUSH = 8

STRENGTH_LABELS = [
    (90, "Very Strong"),
    (80, "Strong"),
    (70, "Solid"),
    (60, "Above Average"),
    (40, "Average"),
    (30, "Below Average"),
    (20, "Weak"),
    (10, "Extremely Weak"),
    (0,  "Just Fold"),
]

# Number of simulations run in main.
MC_SIMULATIONS = 10000

newDeck = [
    '10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
    '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
    '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
    'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
    'QC', 'QD', 'QH', 'QS'
]

def parse_card(card: str):
    """Return (rank_int, suit_char) for a card string like '10H' or 'AS'."""
    return CARD_VALUE[card[:-1]], card[-1]


def best_five_from(cards):
    """Given 2–7 cards, return the best possible 5-card hand as a comparable tuple."""
    parsed = [parse_card(c) for c in cards]

    if len(parsed) <= 5:
        combos = [parsed]
    else:
        combos = list(combinations(parsed, 5))

    best = None
    for hand in combos:
        score = _score_five(hand)
        if best is None or score > best:
            best = score
    return best


def _score_five(hand):
    """Score exactly 5 cards.  Returns a directly comparable tuple. """
    ranks = sorted([r for r, _ in hand], reverse=True)
    suits = [s for _, s in hand]

    is_flush = len(set(suits)) == 1

    # Check for straight
    is_straight, straight_high = check_straight(ranks)

    # Group by rank
    rank_counts = Counter(ranks)

    groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    groups_by_count = sorted(groups, key=lambda x: (x[1], x[0]), reverse=True)
    counts = sorted(rank_counts.values(), reverse=True)

    # Straight flush
    if is_flush and is_straight:
        return (STRAIGHT_FLUSH, straight_high)

    # Quads
    if counts[0] == 4:
        quad_rank = ranks_with_count(rank_counts, 4)[0]
        kicker    = ranks_with_count(rank_counts, 1)[0]
        return (FOUR_OF_KIND, quad_rank, kicker)

    #Full house
    if counts[0] == 3 and counts[1] == 2:
        trips_rank = ranks_with_count(rank_counts, 3)[0]
        pair_rank  = ranks_with_count(rank_counts, 2)[0]
        return (FULL_HOUSE, trips_rank, pair_rank)

    # Flush
    if is_flush:
        return (FLUSH, *ranks)  # ranks already sorted desc

    # Straight
    if is_straight:
        return (STRAIGHT, straight_high)

    # Trips
    if counts[0] == 3:
        trips_rank = ranks_with_count(rank_counts, 3)[0]
        kickers    = sorted(ranks_with_count(rank_counts, 1), reverse=True)
        return (THREE_OF_KIND, trips_rank, *kickers[:2])

    # 2 pair
    if counts[0] == 2 and counts[1] == 2:
        pairs   = sorted(ranks_with_count(rank_counts, 2), reverse=True)
        kickers = ranks_with_count(rank_counts, 1)
        return (TWO_PAIR, pairs[0], pairs[1], kickers[0])

    # Pair
    if counts[0] == 2:
        pair_rank = ranks_with_count(rank_counts, 2)[0]
        kickers   = sorted(ranks_with_count(rank_counts, 1), reverse=True)
        return (PAIR, pair_rank, *kickers[:3])

    # High card
    return (HIGH_CARD, *ranks)


def ranks_with_count(rank_counts, n):
    """Return list of ranks that appear exactly n times, sorted descending."""
    return sorted([r for r, c in rank_counts.items() if c == n], reverse=True)


def check_straight(ranks):
    """
    Given 5 ranks (sorted desc), return (is_straight, high_card).
    Handles the A-2-3-4-5 wheel (high_card = 5 in that case).
    """
    unique = sorted(set(ranks), reverse=True)
    if len(unique) < 5:
        return False, 0

    # Normal straight
    if unique[0] - unique[4] == 4:
        return True, unique[0]

    # Ace, 2, 3, 4 & 5 Straight
    if set(unique) == {14, 2, 3, 4, 5}:
        return True, 5

    return False, 0



def calculate_hand_ranking(cards, isOwnHand):
    """Wraps best_five_from to return with [] for now."""
    score = best_five_from(cards)
    return score, []

def get_hand_ranking(playersHand, sharedCards):
    """Returns how strong the player's hand is compared to all other possible hands."""
    revealedCards = playersHand + sharedCards

    if len(revealedCards) >= 2:
        ahead = tied = behind = 0
        playersRank, potentialHands = calculate_hand_ranking(revealedCards, True)

        remaining = [c for c in FULL_DECK if c not in revealedCards]
        for combination in combinations(remaining, 2):
            possibility = sharedCards + list(combination)
            opponentsHand, _ = calculate_hand_ranking(possibility, False)
            if playersRank > opponentsHand:
                ahead += 1
            elif playersRank == opponentsHand:
                tied += 1
            else:
                behind += 1

        value = ((ahead + tied / 2) / (ahead + tied + behind)) * 100
        hand_strength = next(label for threshold, label in STRENGTH_LABELS if value >= threshold)
        return value, hand_strength, potentialHands


def monte_carlo_equity(players_hole_cards, community_cards, simulations=10000):
    """Simluate *simulations* number of times to calculate equity."""
    all_known_cards = community_cards + [card for hand in players_hole_cards for card in hand]
    remaining_deck  = [c for c in newDeck if c not in all_known_cards]
    cards_needed    = 5 - len(community_cards)
    num_players     = len(players_hole_cards)

    equity_points = [0.0] * num_players
    wins          = [0]   * num_players
    ties          = [0]   * num_players
    losses        = [0]   * num_players

    for _ in range(simulations):
        sampled    = random.sample(remaining_deck, cards_needed)
        full_board = community_cards + sampled
        scores     = [calculate_hand_ranking(h + full_board, False)[0]
                      for h in players_hole_cards]
        best_score = max(scores)
        winners    = [i for i, s in enumerate(scores) if s == best_score]
        share      = 1.0 / len(winners)

        for i in range(num_players):
            if i in winners:
                equity_points[i] += share
                if len(winners) == 1:
                    wins[i] += 1
                else:
                    ties[i] += 1
            else:
                losses[i] += 1

    results = []
    for i in range(num_players):
        equity = round(equity_points[i] / simulations * 100, 2)
        results.append({
            'player':     i + 1,
            'hole_cards': players_hole_cards[i],
            'equity':     equity,
            'wins':       wins[i],
            'ties':       ties[i],
            'losses':     losses[i],
        })
    return results


if __name__ == "__main__":
    mc_start = time.time()
    mc_results = monte_carlo_equity(
        players_hole_cards=[['AH', '2H'], ['QS', 'QC']],
        community_cards=[],
        simulations=MC_SIMULATIONS
    )
    mc_end = time.time()

    print("=== Monte Carlo ===")
    for r in mc_results:
        print(
            f"Player {r['player']} {r['hole_cards']}: {r['equity']}% equity  |  W: {r['wins']} T: {r['ties']} L: {r['losses']}")
    print(f"Time taken: {round(mc_end - mc_start, 2)} seconds")
