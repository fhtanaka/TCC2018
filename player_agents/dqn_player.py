import torch.optim as optim
from network import *
import math

#This player plays based on a Deep Q-learning Network
class dqn_player():
    def __init__(self, config, device):

        self.device=device

        self.board_size = config.board_size
        self.padding = config.padding
        self.color = config.color
        if (self.color == 0):
            self.opponent = 1
        else:
            self.opponent = 0

        self.eps_end = config.eps_end
        self.eps_start = config.eps_start
        self.eps_end = config.eps_end
        self.eps_decay = config.eps_decay
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.target_update = config.target_update

        # This part is for the network
        self.policy_net = DQN(config).to(device)
        self.target_net = DQN(config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(config.replay_memory)
        self.steps_done = 0

    '''
    In the network each space of the board is mapped to an intenger beetwen 0 and board_size^2
    the next 2 functions translates the integer to the index of the board and vice-versa
    '''
    def action_to_index(self, action):
        i = int(action/self.board_size)+self.padding
        j = int(action%self.board_size)+self.padding
        return (i,j)

    def index_to_action(self, index):
        return (index[0]-self.padding)*self.board_size+(index[1]-self.padding)

    '''
    Sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly. 
    The probability of choosing a random action will start at eps_start and will decay exponentially towards eps_end. 
    eps_decay controls the rate of the decay
    '''
    def explore_exploit(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        return sample > eps_threshold

    # Selects an action disregard with its legality
    def select_action(self, state, optimal=False):
        if (self.explore_exploit() or optimal):
            with torch.no_grad():
                net = self.policy_net(state) # Returns the expected value of each action
                return net.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.board_size**2)]], device=self.device, dtype=torch.long)

    # Selects a legal action
    def select_valid_action(self, state, valid_actions, optimal=False):
        valid = list(map(self.index_to_action, valid_actions)) #Converts the indexes (x,y) in actions z
        if (self.explore_exploit() or optimal):
            with torch.no_grad():
                net = self.policy_net(state) # Returns the expected value of each action
                action=valid[net[0][valid].max(0)[1]] # Select the action with max values from the indexes in valid_actions
        else:
            action=random.choice(valid)
        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def play_reward(self, action, state, next_state):
        reward = torch.tensor((0,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def win_reward(self, action, state, next_state):
        reward = torch.tensor((+100,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        self.optimize_model()

    def lose_reward(self, action, state, next_state):
        reward = torch.tensor((-100,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        self.optimize_model()

    def lose_reward_turn_influenced(self, action, state, next_state, turn):
        r = -100 + turn*75/(self.board_size**2)
        reward = torch.tensor((r,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        self.optimize_model()

    def optimize_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
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

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

