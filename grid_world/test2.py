from grid_world import *
from agent import *
from config import *

torch.manual_seed(13)
np.random.seed(13)
config = config()
cpu = grid_agent(config, device)
grid = Grid_world(size = config.board_size, special=config.special, device=device)
print(grid.grid, "\n")

# for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
#     print(param, "\n")

for i in range(0,100):
    file = open("Testes/" + str(i)+".txt", "w")
    grid.reset()

    state = grid.preprocess()
    action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    action = torch.tensor([[1]])
    reward, done = grid.step(action)
    cpu.add_action(state, action, grid.preprocess(), reward)
    cpu.optimize()

    grid.reset()

    state = grid.preprocess()
    action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True, file=file)
    action = torch.tensor([[2]])
    reward, done = grid.step(action)
    cpu.add_action(state, action, grid.preprocess(), reward)
    cpu.optimize()



# for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
#     print(param, "\n")

action = cpu.select_valid_action(state, grid.possible_actions(), optimal = True)