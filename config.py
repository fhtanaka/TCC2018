class config():
    # game config
    color = "black"
    board_size = 6

    # netowrk config 
    height = board_size
    width = board_size
    channels = 2
    classes = 2
    conv_layers = [2, 4] # Size of convolutional layers
    kernel= [3, 2] # Size of the kernel in the conv layers
    #pool = [1, 1] # Size of the pooling
    nn_layers = [8] # Size of the neural netowrk layers
    batch_size = 16
    learning_rate = 0.0054
    momentum = 0.1
    color = color
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.075
    EPS_DECAY = 200