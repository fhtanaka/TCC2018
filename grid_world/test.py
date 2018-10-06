from grid_world import *
from agent import *
from config import *

config = config()
num_episodes = 1000

if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = grid_agent(config, device)
print("Beginning training of ", num_episodes, " episodes")
step = 0
max_turns = (config.board_size**3)
grid = Grid_world(size = config.board_size, special=config.special, device=device)
for i in range(num_episodes):
    grid.reset()
    done = False
    acc_reward = 0
    turns= 0
    file = open("Testes/" + str(i)+".txt", "w") 
    while (done == False and turns < max_turns):
        state = grid.preprocess()
        file.write(str(state) + "\n")
        action = cpu.select_valid_action(state, grid.possible_actions(), file)
        reward, done = grid.step(action)
        acc_reward += reward
        next_state = grid.preprocess()

        cpu.add_action(state, action, next_state, reward)

        if (step%config.optimize == 0):
            cpu.optimize()
        if (step%config.update_target_net == 0):
            cpu.update_target_net()

        turns +=1
        step += 1
    file.close
    print("(",i,")\t\t reward: ", acc_reward , "\t\t steps: ", turns)