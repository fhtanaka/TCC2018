from grid_world import *
from agent import *
import torch
# 13 é do lado do objetivo
# 5 é dois epsços de distancia
seed = 5
torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class config ():
    # game config
    def __init__(self):
        self.board_size = 5
        self.special = 2
        self.padding = 0 # size of padding for neurohex_client
        # netowrk config
        self.channels = 1
        self.conv_layers = [] # Size of convolutional layers
        self.kernel= [] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.batch_size = 1
        self.replay_memory = 1
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.4
        self.eps_decay = 200

        self.momentum = 0.1
        self.lr = 0.01

        self.optimize = 1
        self.update_target_net= 1000

config = config()
cpu = grid_agent(config, device)
grid = Grid_world(size = config.board_size, special=config.special, device=device)
print(grid.grid, "\n")

# for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
#     print(param, "\n")
for i in range(0,1000):
    file = open("Testes/" + str(i)+".txt", "w")
    grid.reset()

    state = grid.preprocess()
    action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    action = torch.tensor([[2]])
    reward, done = grid.step(action)
    cpu.add_action(state, action, grid.preprocess(), reward)
    cpu.optimize()

    state = grid.preprocess()
    action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    action = torch.tensor([[2]])
    reward, done = grid.step(action)
    cpu.add_action(state, action, grid.zero_grid(), reward)
    cpu.optimize()
    grid.reset()



    # state = grid.preprocess()
    # action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    # action = torch.tensor([[1]])
    # reward, done = grid.step(action)
    # cpu.add_action(state, action, grid.preprocess(), reward)
    # cpu.optimize()

    # state = grid.preprocess()
    # action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    # action = torch.tensor([[1]])
    # reward, done = grid.step(action)
    # cpu.add_action(state, action, grid.zero_grid(), -1000)
    # cpu.optimize()

    cpu.update_target_net()


# for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
#     print(param, "\n")

action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True)