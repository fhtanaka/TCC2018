from grid_world import *
from agent import *
from config import *
seed = 6
torch.manual_seed(seed)
np.random.seed(seed)
config = config()

if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = grid_agent(config, device)
print("Beginning training of ", num_episodes, " episodes")
step = 0
max_turns = (15)
grid = Grid_world(size = config.board_size, special=config.special, device=device)
print(grid.grid)
for i in range(num_episodes):
    grid.reset()
    done = False
    acc_reward = 0
    turns= 0
    file = open("Testes/" + str(i)+".txt", "w")
    actions = []
    while (done == False and turns < max_turns):
        state = grid.preprocess()
        
        action = cpu.select_valid_action(state, grid.possible_actions(), file=file)
        reward, done = grid.step(action)
        acc_reward += reward

        if (done):
            next_state = grid.zero_grid()
        else:
            next_state = grid.preprocess()

        actions.append ((state, action, next_state, reward))

        # if (step%config.optimize == 0):
        #     cpu.optimize()
        # if (step%config.update_target_net == 0):
        #     cpu.update_target_net()

        turns +=1
        step += 1
    

    if (done == True):
        for unit in actions:
            state, action, next_state, reward = unit
            cpu.add_action(state, action, next_state, reward)
        cpu.optimize()
        cpu.update_target_net()

        # for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
        #     print(param)
        
    file.close
    print("(",i,")\t\t reward: ", acc_reward , "\t\t steps: ", turns)

grid.reset()
file = open("q_values.txt", "w")
cpu.print_Q_values(cpu, grid, file = file)
file.close
