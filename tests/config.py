import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
white = 0
black = 1

# num_episodes, opponent_method, filename=False, boards_to_print=-1
training_regime = [
    # (100000, "random", 'random.txt', 1000),
    (300000, "eletric", "eletric3.txt", 100)
]


class config ():
    # game config
    def __init__(self, color):
        self.color = color
        self.board_size = 5
        self.padding = 1 # size of padding for neurohex_client
        # netowrk config
        self.channels = 6
        self.conv_layers = [48] # Size of convolutional layers
        self.kernel= [2] # Size of the kernel in the conv layers
        # pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.lr = 0.02
        self.momentum = 0.1

        self.batch_size = 80
        self.replay_memory = 150
        self.policy_net_update = 1
        self.target_net_update = 40
        self.gamma = 0.9
        self.eps_start = 0.9
        self.eps_end = 0.15
        self.eps_decay = 200
