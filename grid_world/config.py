import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_episodes = 10000

class config ():
    # game config
    def __init__(self):
        self.board_size = 5
        self.special = 8
        self.padding = 0 # size of padding for neurohex_client
        # netowrk config
        self.channels = 1
        self.conv_layers = [] # Size of convolutional layers
        self.kernel= [] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.batch_size = 15
        self.replay_memory = 15
        self.gamma = 0.9
        self.eps_start = 0.9
        self.eps_end = 0.4
        self.eps_decay = 200
        self.momentum = 0.1
        self.lr = 0.01

        self.optimize = 1
        self.update_target_net= 1000
