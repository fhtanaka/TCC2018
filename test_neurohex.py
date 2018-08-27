from network import *
from player_agents import *
from neurohex_client import *
# import matplotlib
# import matplotlib.pyplot as plt
def np_to_torch(nparray):
        tensor=torch.from_numpy(nparray)
        tensor = torch.tensor(tensor, device=device, dtype=torch.float)
        return tensor

class config():
    # game config
    color = "white"
    board_size = 5
    padding = 1 # size of padding for neurohex_client
    # netowrk config
    channels = 6
    conv_layers = [12, 12] # Size of convolutional layers
    kernel= [3, 2] # Size of the kernel in the conv layers
    #pool = [1, 1] # Size of the pooling
    nn_layers = [8] # Size of the neural netowrk layers
    batch_size = 16
    # learning_rate = 0.0054
    # momentum = 0.1
    replay_memory=100
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

print("Iniciando")

training_episodes = 10000
test_episodes = 1000
wins = 0
momentum = 0
max_momentum = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

config = config()
cpu = cpu_player(config, device)

for i in range(training_episodes):
    print("training: ", i)
    game = neurohex_game(config.board_size, config.padding)
    turn = 0
    while (game.winner() == None):
        if (turn%2 == 0):
            valid = False
            while not valid:
                state =np_to_torch(game.preprocess())
                action = cpu.select_valid_action(state, game.legal_actions())
                valid, end, reward = cpu.reward(game, action, 1)
                next_state =np_to_torch( game.preprocess()) if valid else None
                cpu.memory.push(state, action, next_state, reward)
                cpu.optimize_model()
        else:
            game.random_play()
        turn += 1

plot = []
for i in range (test_episodes):
    print("test: ", i)
    game = neurohex_game(config.board_size, config.padding)
    turn = 0
    while (game.winner() == None):
        if (turn%2 == 0):
            valid = False
            while not valid:
                state =np_to_torch(game.preprocess())
                action = cpu.select_valid_action(state, game.legal_actions(), optimal=True)
                valid, end, reward = cpu.reward(game, action, 1)
                next_state =np_to_torch( game.preprocess()) if valid else None
                cpu.memory.push(state, action, next_state, reward)
                cpu.optimize_model()
        else:
            game.random_play()
        turn += 1
    print(game)
    if (game.winner()==1):
        wins+=1
        momentum +=1
        if (momentum > max_momentum):
            max_momentum = momentum
    else:
        momentum = 0
    plot.append(wins)

file = open("resultado.txt", "w")
file.write("\nWin percentage: " + str(wins/test_episodes))
file.write("\nMax consecutives wins: " + str(max_momentum))
file.write("\n\n")
file.write(str(plot))
file.write("\n\n")
file.close 
print("Win percentage: " + str(wins/test_episodes))
print("Max consecutives wins: " + str(max_momentum))
# plt.ylim(0, test_episodes)
# plt.plot(plot)
# plt.show()



# tensor([-0.2889, -0.0611], grad_fn=<SelectBackward>)
