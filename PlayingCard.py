from enum import Enum

class Suit(Enum):
    CLUBS = "clubs"
    DIAMONDS = "diamonds"
    HEARTS = "hearts"
    SPADES = "spades"


class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

    @property
    def rank_int(self):
        """Converts suit to integer value"""
        order = [r for r in Rank]
        return order.index(self) + 2


class PlayingCard:
    """
    PlayingCard class is a playing card object.
    Stores the rank & suit.

    card_a = PlayingCard(Rank.*RANK*, Suit.*SUIT*)
    """
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        """
        Prints a card in "RS" (RANK, SUIT) format.
        """
        return f"{self.rank.value}{self.suit.value[0].upper()}"

    def __eq__(self, other):
        if not isinstance(other, PlayingCard):
            return False
        return (self.rank == other.rank) and (self.suit == other.suit)

    def __hash__(self):
        return hash((self.rank, self.suit))


card_a = PlayingCard(Rank.ACE, Suit.CLUBS)
card_b = PlayingCard(Rank.ACE, Suit.CLUBS)
card_c = PlayingCard(Rank.ACE, Suit.SPADES)

hand = set()

hand.add(card_a)
hand.add(card_b)
hand.add(card_c)

print(card_a)
