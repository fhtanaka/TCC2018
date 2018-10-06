import random
import torch
import numpy as np

position = 100
goal = 200
trap = 300

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

class Grid_world():

    def __init__(self, size=10, special=10, device="cpu"):
        
        self.size = size
        self.device = device
        special_spaces = ()
        while (len(special_spaces) < special):
            special_spaces =  totuple(np.unique(np.random.randint(size, size=(special,2)), axis=0))
        self.pos = self.start = special_spaces[0]
        self.end = special_spaces[-1]
        self.traps = special_spaces[1:-1]
        
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)] # translations

        self.reset()


    def reset (self):
        self.grid = np.zeros((self.size, self.size))
        self.pos = self.start
        self.grid[self.start] = position
        self.grid[self.end] = goal
        for i in self.traps:
            self.grid[i] = trap

    def possible_actions (self):
        actions_allowed = []
        x, y = self.pos
        if (x > 0):  # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if (y < self.size - 1):  # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        if (x < self.size - 1):  # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if (y > 0):  # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])

        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed

    def step(self, action):
        # Evolve agent state
        next_pos = (self.pos[0] + self.action_coords[action][0],
                      self.pos[1] + self.action_coords[action][1])

        if (self.grid[next_pos] == goal):
            reward = +100
            done = True
        elif (self.grid[next_pos] == trap):
            reward = -100
            done = False
        else:
            reward = -1
            done = False
        # Terminate if we reach bottom-right grid corner
        self.grid[self.pos] = 0
        self.grid[next_pos] = position
        self.pos = next_pos
        return reward, done

    def preprocess(self):
        tensor=torch.from_numpy(self.grid*1)
        tensor = torch.tensor(tensor, device=self.device, dtype=torch.float)
        
        new_board = torch.zeros((1, 1, self.size, self.size), device=self.device)
        new_board[0] = tensor
        return new_board


