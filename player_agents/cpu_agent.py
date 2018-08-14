import torch.optim as optim
from network import *
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cpu_player():
    def __init__(self, color, board_size, config):

        self.config = config
        self.policy_net = DQN(self.config).to(device)
        self.target_net = DQN(self.config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    def preprocess(self, gamestate):
        size = gamestate.size
        board = torch.from_numpy(gamestate.board)
        # 2 channels, one for each color
        # Neurohex use 6 channels: white pieces, black pieces, white pieces connected to the top,
        # white pieces connected to the bottom, black pieces connected to the left and black pieces
        # connected to the right

        #matrix = np.dstack((board, board))
        matrix = torch.zeros((1, 2, size, size))
        matrix[0][0] = matrix[0][1] = board
        matrix[0][0][matrix[0][0]==2] = 0
        matrix[0][1][matrix[0][0]==1] = 0
        return matrix # Returns here to not add any padding

        # padUD = torch.full((2, 2, size+4), 1)
        # padLR = torch.full((2, size, 2), 2)
        # matrix = torch.cat((matrix, padLR), 2)
        # matrix = torch.cat((padLR, matrix), 2)
        # matrix = torch.cat((matrix, padUD), 1)
        # matrix = torch.cat((padUD, matrix), 1)
        # aux = torch.empty(1,2,size+4, size+4)
        # aux[0] = matrix
        # return aux 

    # sometimes use our model for choosing the action, and sometimes we’ll just sample one uniformly. 
    # The probability of choosing a random action will start at EPS_START and will decay exponentially towards EPS_END. 
    # EPS_DECAY controls the rate of the decay
    def select_action(self, state, optimal=False):
        sample = random.random()
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * math.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or optimal:
            with torch.no_grad():
                return self.policy_net(self.preprocess(state)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.config.board_size**2)]], device=device, dtype=torch.long)



    def optimize_model(self):
        if len(self.memory) < self.config.batch_size:
            return
        transitions = self.memory.sample(self.config.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)

        if [s for s in batch.next_state if s is not None] == []:
            return
        else:
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.config.batch_size, device=device)
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

    def action_to_play(self, action):
        i = int(action/self.config.board_size)
        j = int(action%self.config.board_size)
        return (i,j)


    # return valid, end, reward
    def reward(self, gamestate, action, player):
        try:
            gamestate.play(self.action_to_play(action))
            if (gamestate.winner() == 0):
                return (True, False, torch.tensor((+1,)))
            if(gamestate.winner() == player):
                return (True, True, torch.tensor((+150,)))
            else:
                return (True, True, torch.tensor((-150,)))

        except ValueError:
            return (False, False, torch.tensor((-100,))) 