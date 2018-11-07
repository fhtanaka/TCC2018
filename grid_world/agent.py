from DQN import *
from memory import *
import numpy as np
import torch.optim as optim
import math

action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
actions=["up", "right", "down", "left"]

class grid_agent:

    def __init__(self, config, device, model=False):

        self.device=device

        self.board_size = config.board_size

        self.eps_end = config.eps_end
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.gamma = config.gamma
        self.batch_size = config.batch_size

        # This part is for the network
        if (model != False):
            self.policy_net = torch.load(model)
        else:
            self.policy_net = DQN(config).to(device)
        
        # Be aware that the config must be the exact same for the loaded model
        self.target_net = DQN(config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), momentum=config.momentum, lr=config.lr)
        self.criterion = torch.nn.SmoothL1Loss()
        self.memory = ReplayMemory(config.replay_memory)
        self.steps_done = 0

    def explore_exploit(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # eps_threshold = self.eps_end
        return sample > eps_threshold

    def select_valid_action(self, state, valid, file = False, optimal = False):
        
        net = self.policy_net(state) # Returns the expected value of each action
        
        if (file):
            file.write(str(state) + "\n")
            for i in valid:
                file.write(str(actions[i]) + ": " + str(round(float(net[0][i].data), 4)) + "\n")
        
        if (self.explore_exploit() or optimal == True):
            with torch.no_grad():
                # print(net)
                action=valid[net[0][valid].max(0)[1]] # Select the action with max values from the indexes in valid_actions
        else:
            if (file):
                file.write("Random ")
            action=random.choice(valid)
        
        if (file):
            file.write("Action: " +  str(action) + "\n\n")
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def add_action(self, state, action, next_state, reward):
        r = torch.tensor((reward,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, r)

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states  = [s for s in batch.next_state if s is not None]
        if non_final_next_states  == []:
            return
        else:
            non_final_next_states = torch.cat(non_final_next_states )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        # the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()
        # Compute Huber loss
        # print(reward_batch.float())
        # print(next_state_values )
        # print(expected_state_action_values)
        # print(batch.next_state)
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # print("!"*50)
        # print(reward_batch.data[0])
        # print(loss.data[0])

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def print_Q_values(self, cpu, board, file=False):
        grid = board.grid
        grid[grid==100] = 0

        for i, j in np.transpose(np.where(grid == 0)):
            grid[i][j] = 100
            board.pos = (i, j)
            cpu.select_valid_action(board.preprocess(), board.possible_actions(), file=file)

            grid[i][j] = 0

        board.reset()