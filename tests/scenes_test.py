import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from tqdm import tqdm

seed=1
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.conv_layers = [6] # Size of convolutional layers
        self.kernel= [1] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.lr = 0.01
        self.momentum = 0.1

        self.batch_size = 1
        self.replay_memory = 1
        self.policy_net_update = 1
        self.target_net_update = 20
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200


game = hex_game(5,1, device=device)
game.play("a1")
game.play("b5")
game.play("a2")
game.play("c5")
game.play("a3")
game.play("d5")
game.play("a4")
game.play("e5")
scenario1 = (torch.tensor(game.super_board), "a5")

game = hex_game(5,1, device=device)
game.play("c1")
game.play("b3")
game.play("c2")
game.play("d3")
game.play("c4")
game.play("e3")
game.play("c5")
game.play("a3")
scenario2 = (torch.tensor(game.super_board), "c3")

game = hex_game(5,1, device=device)
game.play("d1")
game.play("a4")
game.play("d2")
game.play("c4")
game.play("c3")
game.play("e5")
game.play("b3")
game.play("d5")
game.play("b5")
game.play("d4")
scenario3 = (torch.tensor(game.super_board), "b4")

game = hex_game(5,1, device=device)
game.play("e5")
game.play("b1")
game.play("e2")
game.play("c1")
game.play("e3")
game.play("d1")
game.play("e4")
game.play("a1")
scenario4 = (torch.tensor(game.super_board), "e1")

scenarios = [scenario1, scenario4]
color = white
print(device, "\n")
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True
cpu = dqn_player(config(white), device)

if __name__ == "__main__":
    for i in range (1000):
        scene = random.choice(scenarios)
        game = hex_game(5,1, device=device, board = scene[0])

        state = torch.tensor(game.super_board)
        action = cpu.select_valid_action(game)
        game.play(game.action_to_index(action))
        next_state = torch.tensor(game.super_board)

        if (game.winner() != None):
            cpu.win_reward(action, state, next_state)
        else:
            op_state = torch.tensor(game.mirror_board())
            op_action = torch.tensor([[game.notation_to_action(scene[1])]], device=device, dtype=torch.long)
            game.play(scene[1])
            op_next_state = torch.tensor(game.mirror_board())

            cpu.lose_reward(action, state, next_state)
            # cpu.win_reward(op_action, op_state, op_next_state)

        if (i%cpu.policy_net_update == 0):
            cpu.optimize_policy_net()

        if (i%cpu.target_net_update == 0):
            cpu.optimize_target_net()
        print(game)