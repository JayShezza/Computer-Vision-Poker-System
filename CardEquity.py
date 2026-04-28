from itertools import combinations
from collections import Counter
import time
import random

CARD_VALUE = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12,'A': 13}

FULL_DECK = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S', '4C', '4D', '4H', '4S',
            '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S', '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S',
            '9C', '9D', '9H', '9S', 'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
            'QC', 'QD', 'QH', 'QS']

HAND_VALUES = {
    'high_card':      0,
    'pair':           13 * 1,
    'two_pair':       13 * 2,
    'three_of_kind':  13 * 3,
    'straight':       13 * 4,
    'flush':          13 * 5,
    'full_house':     13 * 6,
    'four_of_kind':   13 * 7,
    'straight_flush': 13 * 8,
}

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

MC_SIMULATIONS = 50000


def check_pairing(cards, isOwnHand):
    """
    Checks if the cards can make up any type of pairing,
    Modifies potential hands if it is possible for the player to make any higher pairing,
    returns the hand strenght.
    """
    potentialHands = []
    numberOfCardValues = Counter(value for value, _ in cards)
    maxNumber = secondMax = 0
    maxNumberKey = 0

    for key, value in numberOfCardValues.items():
        if value > maxNumber and key > maxNumberKey:
            secondMax = maxNumber
            maxNumber = value
            maxNumberKey = key

        elif value > secondMax:
            secondMax = value
            if secondMax == maxNumber and key > maxNumberKey:
                maxNumberKey = key

    # Four of a kind
    if maxNumber >= 4:
        return maxNumberKey + HAND_VALUES['four_of_kind'], potentialHands

    # Full House
    elif maxNumber >= 3 and secondMax >= 2:
        if 7 - len(cards) + 3 >= 4 and isOwnHand:
            potentialHands.append("Four of a kind")
        return maxNumberKey + HAND_VALUES['full_house'], potentialHands

    # Three of a kind
    elif maxNumber == 3:
        if 7 - len(cards) + 3 >= 4 and isOwnHand:
            potentialHands.append("Four of a kind")
        if 7 - len(cards) + 3 >= 5 and isOwnHand:
            potentialHands.append("Full house")
        return maxNumberKey + HAND_VALUES['three_of_kind'], potentialHands

    # Two pair
    elif maxNumber == 2 and secondMax == 2:
        if 7 - len(cards) + 4 >= 5 and isOwnHand:
            potentialHands.append("Full house")
        if 7 - len(cards) + 2 >= 3 and isOwnHand:
            potentialHands.append("Three of a kind")
        return maxNumberKey + HAND_VALUES['two_pair'], potentialHands

    # Pair
    elif maxNumber == 2:
        if 7 - len(cards) + 2 >= 3 and isOwnHand:
            potentialHands.append("Three of a kind")
            potentialHands.append("Two pair")
        return maxNumberKey + HAND_VALUES['pair'], potentialHands

    # High Card
    if 7 - len(cards) + 1 >= 2 and isOwnHand:
        potentialHands.append("Pair")

    return max(value for value, _ in cards), potentialHands


def check_straight_or_flush(cards, isOwnHand):
    """
    Checks if the cards can make up a flush of a straight,
    Modifies potential hands if it is possible for the player to make a flush or straight with the remaining hands,
    returns the hand strenght.
    """
    potentialHands = []
    sortedCards = sorted(cards, key=lambda x: x[0])
    suiteCount = {'C': 0, 'S': 0, 'H': 0, 'D': 0}

    for _, letter in sortedCards:
        suiteCount[letter] += 1

    isFlush = None
    inARow = []
    potentialInARow = []
    doNotAdd = False

    for i in range(1, len(sortedCards)):
        if sortedCards[i][0] == sortedCards[i - 1][0] + 1 and not doNotAdd:
            if len(inARow) == 0:
                inARow.append(sortedCards[i - 1])
                inARow.append(sortedCards[i])
                potentialInARow.append(sortedCards[i - 1])
                potentialInARow.append(sortedCards[i])

            elif len(inARow) >= 5:
                inARow.pop(0)
                inARow.append(sortedCards[i])

            else:
                inARow.append(sortedCards[i])
                potentialInARow.append(sortedCards[i])

        elif sortedCards[i][0] != sortedCards[i - 1][0]:
            if len(inARow) != 5:
                inARow = []
            else:
                doNotAdd = True

            if isOwnHand and len(cards) > 4:
                addedPot = False
                resetLoop = False
                skipped = 0
                for y in range(7 - len(cards) - skipped):
                    if sortedCards[i][0] == sortedCards[i - 1][0] + 2 + y:
                        addedPot = True
                        skipped += 1
                        y = 7 - len(cards) - skipped

                        if len(potentialInARow) == 0:
                            potentialInARow.append(sortedCards[i - 1])
                            potentialInARow.append(sortedCards[i])
                        else:
                            potentialInARow.append(sortedCards[i])

                    if not addedPot and len(potentialInARow) + 7 - len(cards) < 5 and not resetLoop:
                        potentialInARow = []
                        skipped = 0
                        y = 1
                        resetLoop = True

    if 7 - len(cards) == 0:
        potentialInARow = []

    for key, value in suiteCount.items():
        if value >= 5:
            isFlush = key
            if 'Flush' in potentialHands:
                potentialHands.remove('Flush')

        elif 7 - len(cards) + value >= 5 and isOwnHand and len(cards) > 2 and 'Flush' not in potentialHands:
            potentialHands.append('Flush')

    if (7 - len(cards) + len(potentialInARow) >= 5 and isOwnHand and len(cards) > 2 and len(inARow) < 5):
        potentialHands.append('Straight')

    # Straight Flush
    if (len(inARow) == 5 and isFlush != None):
        numInRow = sum(
            1 for value1, _ in inARow for value2, suite2 in sortedCards if value1 == value2 and suite2 == isFlush)
        if numInRow == 5:
            if all(item[0] == num for item, num in zip(inARow, [9, 10, 11, 12, 13])):
                return 130, potentialHands
            return inARow[-1][0] + HAND_VALUES['straight_flush'], potentialHands

    # Flush
    if (isFlush):
        return max(value for value, suite in sortedCards if suite == isFlush) + HAND_VALUES['flush'], potentialHands

    # Straight
    elif (len(inARow) == 5):
        return inARow[-1][0] + HAND_VALUES['straight'], potentialHands

    return 0, potentialHands


def calculate_hand_ranking(cards, isOwnHand):
    """
    Calculates the hand rankings based of cards given,
    if is a players hand then adds potential hands.
    """
    potentialHands = []
    cardValSuite = []

    for card in cards:
        cardValSuite.append((CARD_VALUE[card[:-1]], card[-1:]))
    possiblePairingVal, pairingPotentialHands = check_pairing(cardValSuite, isOwnHand)
    possibleStraightOrFlushVal, flushStraightPotentialHands = check_straight_or_flush(cardValSuite, isOwnHand)

    if possiblePairingVal < 14 and possibleStraightOrFlushVal > 0:
        potentialHands = flushStraightPotentialHands

    elif len(cards) >= 5:
        potentialHands = pairingPotentialHands + flushStraightPotentialHands

    else:
        potentialHands = pairingPotentialHands

    return max(possiblePairingVal, possibleStraightOrFlushVal), potentialHands


def get_hand_ranking(playersHand, sharedCards):
    """
    Returns how strong the players hand it compared to all other possible hands
    """
    revealedCards = playersHand + sharedCards

    if len(revealedCards) >= 2:
        ahead = 0
        tied = 0
        behind = 0
        playersRank, potentialHands = calculateHandRanking(revealedCards, True)

        for card in revealedCards:
            FULL_DECK.remove(card)
        combinationOfTwo = list(combinations(FULL_DECK, 2))
        for combination in combinationOfTwo:
            possibility = sharedCards + list(combination)
            opponentsHand, _ = calculateHandRanking(possibility, False)
            if playersRank > opponentsHand:
                ahead += 1
            elif playersRank == opponentsHand:
                tied += 1
            else:
                behind += 1
        value = ((ahead + tied / 2) / (ahead + tied + behind)) * 100
        hand_strength = "Just Fold"

        hand_strength = next(label for threshold, label in STRENGTH_LABELS if value >= threshold)

newDeck = ['10C', '10D', '10H', '10S', '2C', '2D', '2H', '2S', '3C', '3D', '3H', '3S',
        '4C', '4D', '4H', '4S', '5C', '5D', '5H', '5S', '6C', '6D', '6H', '6S',
        '7C', '7D', '7H', '7S', '8C', '8D', '8H', '8S', '9C', '9D', '9H', '9S',
        'AC', 'AD', 'AH', 'AS', 'JC', 'JD', 'JH', 'JS', 'KC', 'KD', 'KH', 'KS',
        'QC', 'QD', 'QH', 'QS']


def exact_equity_multiplayer(players_hole_cards, community_cards):
    """
    players_hole_cards: list of lists e.g. [['AC', 'AS'], ['KH', 'KD'], ['QC', 'QD']]
    community_cards: list e.g. ['2H', '7D', '10C']
    NOTE:
    num of simulations depends on how many players and community cards but 2 players no community cards is around 1,700,000
    """
    all_known_cards = community_cards + [card for hand in players_hole_cards for card in hand]
    remaining_deck = [c for c in newDeck if c not in all_known_cards]
    cards_needed = 5 - len(community_cards)

    num_players = len(players_hole_cards)
    wins = [0] * num_players
    ties = [0] * num_players
    losses = [0] * num_players

    for board_completion in combinations(remaining_deck, cards_needed):
        full_board = community_cards + list(board_completion)

        # Score every player's hand on this board
        scores = []
        for hole_cards in players_hole_cards:
            score, _ = calculate_hand_ranking(hole_cards + full_board, False)
            scores.append(score)

        best_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best_score]

        for i in range(num_players):
            if scores[i] == best_score and len(winners) == 1:
                wins[i] += 1
            elif scores[i] == best_score and len(winners) > 1:
                ties[i] += 1
            else:
                losses[i] += 1

    total = sum(wins) + sum(ties) + sum(losses)
    results = []
    for i in range(num_players):
        equity = round((wins[i] + ties[i] / 2) / (wins[i] + ties[i] + losses[i]) * 100, 2)
        results.append({
            'player': i + 1,
            'hole_cards': players_hole_cards[i],
            'equity': equity,
            'wins': wins[i],
            'ties': ties[i],
            'losses': losses[i]
        })

    return results



def monte_carlo_equity(players_hole_cards, community_cards, simulations=100000):
    """
    players_hole_cards: list of lists e.g. [['AC', 'AS'], ['KH', 'KD']]
    community_cards: list e.g. ['2H', '7D', '10C']
    simulations: number of random runouts to sample (higher = more accurate but slower)
    """
    all_known_cards = community_cards + [card for hand in players_hole_cards for card in hand]
    remaining_deck = [c for c in newDeck if c not in all_known_cards]
    cards_needed = 5 - len(community_cards)

    num_players = len(players_hole_cards)
    wins = [0] * num_players
    ties = [0] * num_players
    losses = [0] * num_players

    for _ in range(simulations):
        # Randomly sample just enough cards to complete the board
        sampled = random.sample(remaining_deck, cards_needed)
        full_board = community_cards + sampled

        # Score every player on this random board
        scores = []
        for hole_cards in players_hole_cards:
            score, _ = calculate_hand_ranking(hole_cards + full_board, False)
            scores.append(score)

        best_score = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best_score]

        for i in range(num_players):
            if scores[i] == best_score and len(winners) == 1:
                wins[i] += 1
            elif scores[i] == best_score and len(winners) > 1:
                ties[i] += 1
            else:
                losses[i] += 1

    results = []
    for i in range(num_players):
        total = wins[i] + ties[i] + losses[i]
        equity = round((wins[i] + ties[i] / 2) / total * 100, 2)
        results.append({
            'player': i + 1,
            'hole_cards': players_hole_cards[i],
            'equity': equity,
            'wins': wins[i],
            'ties': ties[i],
            'losses': losses[i]
        })

    return results


# Compare both methods
mc_start = time.time()
mc_results = monte_carlo_equity(
    players_hole_cards = [['3C', '2C'], ['QC', 'KC']],
    community_cards = [],
    simulations = MC_SIMULATIONS
)
mc_end = time.time()

ee_start = time.time()
ee_results = exact_equity_multiplayer(
    players_hole_cards = [['3C', '2C'], ['QC', 'KC']],
    community_cards=[]
)
ee_end = time.time()

print("=== Monte Carlo ===")
for r in mc_results:
    print(
        f"Player {r['player']} {r['hole_cards']}: {r['equity']}% equity  |  W: {r['wins']} T: {r['ties']} L: {r['losses']}")
print(f"Time taken: {round(mc_end - mc_start, 2)} seconds")

print("=== Exact Equity ===")
for r in ee_results:
    print(
        f"Player {r['player']} {r['hole_cards']}: {r['equity']}% equity  |  W: {r['wins']} T: {r['ties']} L: {r['losses']}")
print(f"Time taken: {round(ee_end - ee_start, 2)} seconds")