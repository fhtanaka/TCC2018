import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from gamestate import gamestate



is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if (len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(2592, 25)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class cpu_player():
    def __init__(self, color, board_size):
        self.board_size = board_size
        self.color = color
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
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
        matrix = torch.zeros((2, size, size))
        matrix[0] = matrix[1] = board
        matrix[0][matrix[0]==2] = 0
        matrix[1][matrix[0]==1] = 0
        padUD = torch.full((2, 2, size+4), 1)
        padLR = torch.full((2, size, 2), 2)
        matrix = torch.cat((matrix, padLR), 2)
        matrix = torch.cat((padLR, matrix), 2)
        matrix = torch.cat((matrix, padUD), 1)
        matrix = torch.cat((padUD, matrix), 1)
        aux = torch.empty(1,2,size+4, size+4)
        aux[0] = matrix
        return aux

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(self.preprocess(state)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(25)]], device=device, dtype=torch.long)



    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.float()

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def action_to_play(self, action):
        i = int(action/self.board_size)
        j = int(action%self.board_size)
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

def random_play(gamestate):
    while True:
        try:
            i, j = random.choice(np.argwhere(gamestate.board==0))
            gamestate.play((i,j))
            return True
        except ValueError:
            print("ERRO")
            return False


def test():
    print("Iniciando")
    num_episodes = 10000
    board_size = 5
    cpuA = cpu_player("black", board_size)
    wins = 0
    momentum = 0
    max_momentum = 0
    plot = []
    for i in range (num_episodes):
        print(i)
        game = gamestate(board_size)
        turn = 0
        while (game.moves() != [] and game.winner() == 0):
            if (turn%2 == 0):
                valid = False
                while not valid:
                    state = cpuA.preprocess(game)
                    action = cpuA.select_action(game)
                    valid, end, reward = cpuA.reward(game, action, 1)
                    next_state = cpuA.preprocess(game) if valid else None
                    cpuA.memory.push(state, action, next_state, reward)
                    cpuA.optimize_model()
            else:
                random_play(game)
            turn += 1
        if (game.winner()==1):
            wins+=1
            momentum +=1
            if (momentum > max_momentum):
                max_momentum = momentum
        else:
            momentum = 0
        plot.append(wins)

    file = open("resultado.txt", "w")
    file.write("\nWin percentage: " + str(wins/num_episodes))
    file.write("\nMax consecutives wins: " + str(max_momentum))
    file.write("\n\n")
    file.write(str(plot))
    file.write("\n\n")
    file.close 
    print("Win percentage: " + str(wins/num_episodes))
    print("Max consecutives wins: " + str(max_momentum))
    plt.ylim(0, num_episodes)
    plt.plot(plot)
    plt.show()


test()



