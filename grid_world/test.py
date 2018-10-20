from grid_world import *
from agent import *
from config import *
# torch.manual_seed(1)
# np.random.seed(1)
config = config()
num_episodes = 10000

if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = grid_agent(config, device)
print("Beginning training of ", num_episodes, " episodes")
step = 0
max_turns = (100)
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
            next_state = None
        else:
            next_state = grid.preprocess()

        actions.append ((state, action, next_state, reward))

        if (step%config.optimize == 0):
            cpu.optimize()
        if (step%config.update_target_net == 0):
            cpu.update_target_net()

        turns +=1
        step += 1
    

    if (done == True):
        for unit in actions:
            state, action, next_state, reward = unit
            cpu.add_action(state, action, next_state, reward)
        # for param in cpu.policy_net._modules["full_connected_layer1"].parameters():
        #     print(param)
        
    file.close
    print("(",i,")\t\t reward: ", acc_reward , "\t\t steps: ", turns)
