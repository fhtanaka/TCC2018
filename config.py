import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_regime = [
    (1000, "random", "random.txt", 10),
    (100000, "eletric", "eletric.txt", 100)
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
    kernel= [2, 2] # Size of the kernel in the conv layers
    #pool = [1, 1] # Size of the pooling
    # nn_layers = [8] # Size of the neural netowrk layers
    batch_size = 32
    target_update = 10
    replay_memory=100
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
