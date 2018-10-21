import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from tqdm import tqdm

seed = 2
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
white = 0
black = 1
class config ():
    # game config
    def __init__(self, color):
        self.color = color
        self.board_size = 5
        self.padding = 1 # size of padding for neurohex_client
        # netowrk config
        self.channels = 6
        self.conv_layers = [24] # Size of convolutional layers
        self.kernel= [2] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.batch_size = 1
        self.replay_memory = 1
        self.policy_net_update = 1
        self.target_net_update = 3
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.15
        self.eps_decay = 200

color = white
config = config(color)

game = hex_game(config.board_size, config.padding, device)
game.play("c1")
game.play("a3")
game.play("c2")
game.play("b3")
game.play("c4")
game.play("d3")
game.play("c5")
game.play("e3")
board = torch.tensor(game.board)

cpu = dqn_player(config, device)
for i in range(16):
    game = hex_game(config.board_size, config.padding, board=board)

    state = torch.tensor(game.super_board)
    action = cpu.select_valid_action(game, optimal=True)
    game.play(game.action_to_index(action))
    next_state = torch.tensor(game.super_board)

    if (game.winner() == cpu.color):
        cpu.win_reward(action, state, next_state)
    else:
        game.play("c3")
        cpu.lose_reward(action, state, next_state)
    cpu.optimize_policy_net()
    print(game, "\n")
