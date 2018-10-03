import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
white = 0
black = 1

training_regime = [
    # (100000, "random", 'random.txt', 1000),
    (100000, "mixed", 'mixed.txt', 100)
]


class config ():
    # game config
    def __init__(self, color):
        self.color = color
        self.board_size = 5
        self.padding = 1 # size of padding for neurohex_client
        # netowrk config
        self.channels = 6
        self.conv_layers = [24, 48] # Size of convolutional layers
        self.kernel= [2, 2] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.batch_size = 360
        self.target_update = 25
        self.replay_memory = 1000
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.15
        self.eps_decay = 200
