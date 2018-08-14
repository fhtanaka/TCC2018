class config():
    # game config
    color = "black"
    board_size = 5

    # netowrk config 
    height = board_size
    width = board_size
    channels = 2
    classes = 2
    conv_layers = [16, 16] # Size of convolutional layers
    kernel= [5, 5] # Size of the kernel in the conv layers
    nn_layers = [16, 16] # Size of the neural netowrk layers
    pool = [1, 1] # Size of the pooling
    batch_size = 32
    learning_rate = 0.0054
    momentum = 0.1
    board_size = board_size
    color = color
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.075
    EPS_DECAY = 200