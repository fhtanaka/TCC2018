import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_regime = [
    (100, "eletric", "eletric.txt", 10),
    (100, "random", "random.txt", 10)
]
# test_regime=


class config():
    # game config
    color = 0
    board_size = 5
    padding = 1 # size of padding for neurohex_client
    # netowrk config
    channels = 6
    conv_layers = [12, 24] # Size of convolutional layers
    kernel= [3, 2] # Size of the kernel in the conv layers
    #pool = [1, 1] # Size of the pooling
    # nn_layers = [8] # Size of the neural netowrk layers
    batch_size = 16
    # learning_rate = 0.0054
    # momentum = 0.1
    replay_memory=500
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
