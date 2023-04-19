from typing import Type
import random
import time

# define globals for cards
SUITS = ('C', 'S', 'H', 'D')
RANKS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K')
VALUES = {'A':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, 'T':10, 'J':10, 'Q':10, 'K':10}
INDICES = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12}


# define card class
class Card:
    def __init__(self, suit, rank):
        if (suit in SUITS) and (rank in RANKS):
            self.suit = suit
            self.rank = rank
        else:
            self.suit = None
            self.rank = None
            print(f'Invalid card: {suit} {rank}')

    def __str__(self):
        return self.suit + self.rank

    def get_suit(self):
        return self.suit

    def get_rank(self):
        return self.rank

    def get_value(self):
        return VALUES[self.rank]


# define hand class
class Hand:
    def __init__(self):
        self.cards = []

    def __str__(self):
        ans = "Hand contains "
        for i in range(len(self.cards)):
            ans += str(self.cards[i]) + " "
        return ans
        # return a string representation of a hand

    def add_card(self, card):
        self.cards.append(card)
        # add a card object to a hand
    
    def pop_card(self):
        return self.cards.pop()

    def get_value(self):
        value = 0
        aces = False
        for c in self.cards:
            rank = c.get_rank()
            v = VALUES[rank]
            if rank == 'A': aces = True
            value += v
        if aces and value < 12: value += 10
        return value
        # count aces as 1, if the hand has an ace, then add 10 to hand value if it doesn't bust
        # compute the value of the hand, see Blackjack video


# define deck class
class Deck:
    def __init__(self):
        self.deck = []
        for s in SUITS:
            for r in RANKS:
                self.deck.append(Card(s, r))
        self.hidden_cards = []
        # create a Deck object

    def shuffle(self):
        random.shuffle(self.deck)
        # shuffle the deck

    def deal_card(self):
        return self.deck.pop()
        # deal a card object from the deck

    def __str__(self):
        ans = "The deck: "
        for c in self.deck:
            ans += str(c) + " "
        return ans
        # return a string representing the deck


# define player class
class Player:

    def __init__(self):
        self.matrix = [[[[0,0] for k in range(13)] for j in range(13)] for i in range(13)]
        self.num_trials = 0

    def remove_cards(self, deck, avoid):
        for i in range(len(deck.deck)-1, -1, -1):
            c = deck.deck[i]
            if (c.get_suit(), c.get_rank()) in avoid:
                deck.deck = deck.deck[:i] + deck.deck[i+1:]


    def sim_trials(self, trials, to_hit, i, j, k):
        playerhand = Hand()
        dealerhand = Hand()
        playerhand.add_card(Card(SUITS[0], RANKS[i]))
        playerhand.add_card(Card(SUITS[1], RANKS[j]))
        dealerhand.add_card(Card(SUITS[2], RANKS[k]))
        deck = Deck()
        self.remove_cards(deck, [(0, i), (1, j), (2, k)])
        wins = 0
        for t in range(trials):
            deck.shuffle()
            dealerhand.add_card(deck.deal_card())
            if to_hit:
                playerhand.add_card(deck.deal_card())
                while playerhand.get_value() < 13:
                    playerhand.add_card(deck.deal_card())
            wins += self.winner(playerhand, dealerhand, deck, False)
            deck.deck += dealerhand.cards[1:]
            dealerhand.cards = [dealerhand.cards[0]]
            if to_hit:
                while len(playerhand.cards) > 2:
                    deck.deck.append(playerhand.cards[-1])
                    playerhand.cards.pop()
        return wins/trials

    def sim(self, trials: int) -> None:
        trials = max(10, trials//(13*13*13*2))
        self.num_trials = trials
        # trials //= 13*13*2
        self.matrix = [[[[0,0] for k in range(13)] for j in range(13)] for i in range(13)]
        # For each starting state, we get proportion of wins from hitting
        for i in range(13):
            for j in range(13):
                for k in range(13):
                    self.matrix[i][j][k][0] = self.sim_trials(trials, True, i, j, k)
        # And now same thing, but wins from standing
        for i in range(13):
            for j in range(13):
                for k in range(13):
                    self.matrix[i][j][k][1] = self.sim_trials(trials, False, i, j, k)

    # slight benefit over never hit, ~0.42 at thres=14
    def threshold(self, playerhand, thres):
        if playerhand.get_value() >= thres:
            return False
        return True

    def hitme(self, playerhand: Type[Hand], dealerfacecard: Type[Card]) -> bool:
        if self.num_trials < 35:
            return self.threshold(playerhand, 13)
        if len(playerhand.cards) > 2:
            return self.threshold(playerhand, 13)
        rank1, rank2, rankd = playerhand.cards[0].get_rank(), playerhand.cards[1].get_rank(), dealerfacecard.get_rank()
        mat = self.matrix[INDICES[rank1]][INDICES[rank2]][INDICES[rankd]]
        return mat[0] > mat[1]
        # return self.threshold(playerhand, 21)
        # return False

    def deal(self):
        deck = Deck()
        deck.shuffle()
        playerhand, dealerhand = Hand(), Hand()
        for j in range(2):
            playerhand.add_card(deck.deal_card())
            dealerhand.add_card(deck.deal_card())
        return deck, playerhand, dealerhand

    def winner(self, playerhand, dealerhand, deck, toPlay):
        ret = 0
        # don't ask to hit again if we're checking the winner of a simulation
        if toPlay:
            while self.hitme(playerhand, dealerhand.cards[0]):
                playerhand.add_card(deck.deal_card())
                if playerhand.get_value() > 21:
                    return 0
        # player busts
        if playerhand.get_value() > 21:
            return 0
        # dealer required to hit until >= 17
        while dealerhand.get_value() < 17:
            dealerhand.add_card(deck.deal_card())
        # win condition: dealer busts or player > dealer
        if dealerhand.get_value() > 21 or dealerhand.get_value() < playerhand.get_value():
            return 1
        return 0

    def play(self, trials: int) -> float:
        assert(trials > 0)
        wins = 0
        for t in range(trials):
            deck, playerhand, dealerhand = self.deal()
            wins += self.winner(playerhand, dealerhand, deck, True)
        print(wins, " wins in ", trials, " trials.")
        return wins/trials

p = Player()
s = time.time()
p.sim(35*13*13*13*2)
e = time.time()
print("Time to run simulation: ", e-s)
# print(str(p.matrix))
s = time.time()
print("Proportion of games won: ", p.play(1000_000))
e = time.time()
print("Time to run trials: ", e-s)