import torch.optim as optim
from network import *
import math


#device="cpu"

class cpu_player():
    def __init__(self, config, device):

        self.config = config
        self.policy_net = DQN(self.config).to(device)
        self.target_net = DQN(self.config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(config.replay_memory)
        self.steps_done = 0
        self.device=device

    def explore_exploit(self):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        return sample > eps_threshold

    def np_to_torch(self, nparray):
        tensor=torch.from_numpy(nparray)
        tensor = torch.tensor(tensor, device=self.device, dtype=torch.long)
        return tensor

    # sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly. 
    # The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END. 
    # EPS_DECAY controls the rate of the decay
    def select_action(self, state, optimal=False):
        if self.explore_exploit() or optimal:
            with torch.no_grad():
                net = self.policy_net(state)
                return net.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.config.board_size**2)]], device=self.device, dtype=torch.long)

    def select_valid_action(self, state, valid_actions, optimal=False):
        valid= list(map(self.tuple_to_action, valid_actions))
        if self.explore_exploit() or optimal:
            with torch.no_grad():
                net = self.policy_net(state)
                action=valid[net[0][valid].max(0)[1]] # Select the action with max values from the indexes in valid_actions
        else:
            action=random.choice(valid)
        return torch.tensor([[action]], device=self.device, dtype=torch.long)



    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return
        transitions = self.memory.sample(self.config.batch_size)
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
        next_state_values = torch.zeros(self.config.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.config.GAMMA) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def action_to_tuple(self, action):
        i = int(action/self.config.board_size)+self.config.padding
        j = int(action%self.config.board_size)+self.config.padding
        return (i,j)

    def tuple_to_action(self, tuple):
        return (tuple[0]-self.config.padding)*self.config.board_size+(tuple[1]-self.config.padding)


    # return valid, end, reward
    def reward(self, game, action, player):
        try:
            game.play(self.action_to_tuple(action))
            if (game.winner() == 0):
                return (True, False, torch.tensor((+1,), device=self.device))
            if(game.winner() == player):
                return (True, True, torch.tensor((+150,), device=self.device))
            else:
                return (True, True, torch.tensor((-150,), device=self.device))

        except ValueError:
            return (False, False, torch.tensor((-100,), device=self.device)) 
