from config import config
from network import *
from player_agents import *
from hex_client import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
board_size = 5
config = config()
cpu = cpu_player("white", board_size, config)

game = gamestate(board_size)
board = cpu.preprocess(game)
dqn = DQN(config).to(device)

#print(dqn(board))

print(cpu.select_action(game, optimal=True))