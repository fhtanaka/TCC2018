import random
from game_model import *

class heuristic_player():
    def __init__(self, color):
        self.color = color

    def random_play(self, game):
        return game.tuple_to_action(game.random_play())

    def eletric_resistence_play(self, game):
        values = score(game, self.color)
        action = values.argmax()
        game.play(game.action_to_tuple(action))
        return action
