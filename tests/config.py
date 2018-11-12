import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
white = 0
black = 1

# num_episodes, opponent_method, filename=False, boards_to_print=-1
training_regime = [
    # (150000, "eletric", "workingR.txt", 100),
    (150000, "mixed", "mixed_mirror_2_layers_7_board.txt", 100)
]


class config ():
    # game config
    def __init__(self, 
                color=white,
                board_size=7,
                padding=1,
                conv_layers = [48, 384],
                kernel = [2, 2],
                lr=0.01,
                momentum=0.1,
                batch_size = 200,
                replay_memory = 300,
                policy_net_update = 5,
                target_net_update = 40,
                gamma = 0.8,
                eps_end = 0.1
                ):

        self.color = color
        self.board_size = board_size
        self.padding = padding # size of padding for neurohex_client
        # netowrk config
        self.channels = 6
        self.conv_layers = conv_layers # Size of convolutional layers
        self.kernel= kernel # Size of the kernel in the conv layers
        # pool = [1, 1] # Size of the pooling
        # nn_layers = [8] # Size of the neural netowrk layers
        self.lr = lr
        self.momentum = momentum

        self.batch_size = batch_size
        self.replay_memory = replay_memory
        self.policy_net_update = policy_net_update
        self.target_net_update = target_net_update
        self.gamma = gamma
        self.eps_end = eps_end
        self.eps_start = 0.9
        self.eps_decay = 200

        
