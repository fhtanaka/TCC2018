import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class config ():
    # game config
    def __init__(self):
        self.board_size = 10
        self.special = 25
        self.padding = 0 # size of padding for neurohex_client
        # netowrk config
        self.channels = 1
        self.conv_layers = [1] # Size of convolutional layers
        self.kernel= [1] # Size of the kernel in the conv layers
        #pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.batch_size = 1
        self.replay_memory = 1
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.15
        self.eps_decay = 200

        self.optimize = 10
        self.update_target_net= 1000
