import sys
sys.path.append('../')
from game_model import *

game = hex_game(5,1)
game.play(1)
game.play(4)
game.play(8)
game.play(11)
a = torch.tensor([[1, 0], [1, 1], [1, 3]])
b = torch.tensor([[1, 0], [1, 3], [4, 2]])

print(game)
print(game.legal_indexes())