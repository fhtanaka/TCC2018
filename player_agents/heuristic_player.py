import random
from game_model import *

class heuristic_player():
    def __init__(self, color, strategy, chance=0.):
        self.color = color
        self.chance = chance
        if (strategy == "random"):
            self.play = self.random_play
        elif (strategy == "eletric"):
            self.play = self.eletric_resistence_play
        elif (strategy == "mixed"):
            self.play = self.mixed_play

    def random_play(self, game):
        return game.index_to_action(game.random_play())

    def eletric_resistence_play(self, game):
        values = score(game, self.color)
        action = values.argmax()
        game.play(game.action_to_index(action))
        return action

    def mixed_play(self, game):
        if (random.random() < self.chance):
            return self.eletric_resistence_play(game)
        else:
            return self.random_play(game)