"""
HW 1 - Blackjack
"""
import random

# define globals for cards
SUITS = ("C", "S", "H", "D")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K")
VALUES = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
}


class Card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print("Invalid card: ", suit, rank)

    def __str__(self):
        return self.suit + self.rank

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank


class Hand:
    def __init__(self):
        self.cards = []

    def __str__(self):
        ans = "Hand contains "
        for card in self.cards:
            ans += str(card) + " "
        return ans

    def add_card(self, card):
        self.cards.append(card)

    def get_value(self):
        """
        compute the value of the hand, see Blackjack video
        count aces as 1, if the hand has an ace, then add 10 to hand value if it doesn't bust
        """
        value = 0
        aces = False
        for card in self.cards:
            rank = card.get_rank()
            val = VALUES[rank]
            if rank == "A":
                aces = True
            value += val
        if aces and value < 12:
            value += 10
        return value


class Deck:
    """ define deck class """
    def __init__(self):
        """ create a Deck object """
        self.deck = []
        for s in SUITS:
            for r in RANKS:
                self.deck.append(Card(s, r))

    def shuffle(self):
        random.shuffle(self.deck)

    def deal_card(self):
        return self.deck.pop()

    def __str__(self):
        ans = "The deck: "
        for c in self.deck:
            ans += str(c) + " "
        return ans


# define event handlers for buttons
def deal():
    global theDeck, playerhand, househand
    theDeck = Deck()
    theDeck.shuffle()

    playerhand = Hand()
    househand = Hand()
    playerhand.add_card(theDeck.deal_card())
    playerhand.add_card(theDeck.deal_card())
    househand.add_card(theDeck.deal_card())
    househand.add_card(theDeck.deal_card())


def hit():
    playerhand.add_card(theDeck.deal_card())
    val = playerhand.get_value()
    if val > 21:
        print("You are busted! House wins!")
        return False
    return True
    # if the hand is in play, hit the player
    # if busted, assign a message to OUTCOME, update INPLAY and SCORE


def stand():
    if playerhand.get_value() > 21:
        print("You are busted.")
        return
    val = househand.get_value()
    while val < 17:
        househand.add_card(theDeck.deal_card())
        val = househand.get_value()
    print("Dealer ", househand.__str__())
    if val > 21:
        if playerhand.get_value() > 21:
            print("House is busted, but House wins tie game!")
        else:
            print("House is busted! Player wins!")
    else:
        if val == playerhand.get_value():
            print("House wins ties!")
        elif val > playerhand.get_value():
            print("House wins!")
        else:
            print("Player wins!")

def play_turn():
    print(playerhand.__str__())
    print("Dealer cards: ", househand.cards[0])
    x = input("h to hit, s to stand! ")
    while x not in ['h', 's']:
        x = input("Invalid input. h to hit, s to stand! ")
    if x == 'h':
        if hit():
            play_turn()
    else:
        stand()
        return

if __name__ == "__main__":
    print("Dealing!")
    deal()
    play_turn()