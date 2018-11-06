import torch.optim as optim
from network import *
import math

#This player plays based on a Deep Q-learning Network
class dqn_player():
    def __init__(self, config, device, model=False):

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
        self.policy_net_update = config.policy_net_update
        self.target_net_update = config.target_net_update

        self.policy_loss = 0
        self.target_loss = float("inf")
        self.wins = 0
        self.old_wins = 1

        # This part is for the network
        if (model != False):
            self.policy_net = torch.load(model)
        else:
            self.policy_net = DQN(config).to(device)
        
        # Be aware that the config must be the exact same for the loaded model
        self.target_net = DQN(config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=config.lr, momentum=config.momentum)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=config.lr, momentum=config.momentum)
        self.criterion = torch.nn.SmoothL1Loss()
        self.memory = ReplayMemory(config.replay_memory)
        self.steps_done = 0

        self.reward = 1000

    '''
    Sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly. 
    The probability of choosing a random action will start at eps_start and will decay exponentially towards eps_end. 
    eps_decay controls the rate of the decay
    '''
    def explore_exploit(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        # eps_threshold = self.eps_end
        return sample > eps_threshold

    # Selects an action disregard with its legality
    def select_action(self, game, optimal=False):
        if (self.explore_exploit() or optimal):
            with torch.no_grad():
                net = self.policy_net(game.super_board) # Returns the expected value of each action
                return net.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.board_size**2)]], device=self.device, dtype=torch.long)

    # Selects a legal action
    def select_valid_action(self, game, optimal=False, print_values=False):
        valid = game.legal_actions() #Converts the indexes (x,y) in actions z
        if (self.explore_exploit() or optimal):
            with torch.no_grad():
                net = self.policy_net(game.super_board) # Returns the expected value of each action
                if (print_values):
                    print(net.reshape((self.board_size, self.board_size)))
                action=valid[net[0][valid].max(0)[1]] # Select the action with max values from the indexes in valid_actions
        else:
            action=random.choice(valid)

        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def play_reward(self, action, state, next_state):
        reward = torch.tensor((0,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def win_reward(self, action, state, next_state):
        reward = torch.tensor((self.reward,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def win_reward_turn_influenced(self, action, state, next_state, turn):
        r = self.reward - turn*(self.reward/2)/(self.board_size**2)
        reward = torch.tensor((r,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def lose_reward(self, action, state, next_state):
        reward = torch.tensor((self.reward*-1,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def lose_reward_turn_influenced(self, action, state, next_state, turn):
        r = self.reward*-1 + turn*(self.reward/2)/(self.board_size**2)
        reward = torch.tensor((r,), device=self.device, dtype=torch.long)
        self.memory.push(state, action, next_state, reward)
        # self.optimize_model()

    def optimize_target_net(self):
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        if (self.policy_loss < self.target_loss or self.wins >= self.old_wins):
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_loss = self.policy_loss
            self.policy_loss = 0
            if (self.wins >= self.old_wins): 
                self.old_wins = self.wins
                self.wins = 0
        else:
            self.policy_net.load_state_dict(self.target_net.state_dict())
            self.policy_loss = 0

    def optimize_policy_net(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.float()

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.policy_loss += loss

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

