from DQN import *

class config():
    height = 45
    width = 80
    channels = 3
    classes = 3
    conv_layers = [3, 5, 16]
    nn_layers = [100, 3, 9, 2]
    kernel= [5, 5]
    pool = [1, 1]
    batch_size = 32
    epochs = 1
    learning_rate = 0.0054
    momentum = 0.1
    save_step = 100

conf = config()
dqn = DQN(conf)
x = torch.zeros([config.batch_size, config.channels, config.height, config.width])
print(dqn.forward(x)[0][0][0])
# tensor([-0.2889, -0.0611], grad_fn=<SelectBackward>)
