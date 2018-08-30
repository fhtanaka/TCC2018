from network import *
from player_agents import *
from neurohex_client import *
# import matplotlib
# import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

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
    nn_layers = [8] # Size of the neural netowrk layers
    batch_size = 16
    # learning_rate = 0.0054
    # momentum = 0.1
    replay_memory=100
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200


training_episodes = 100
test_episodes = 10

config = config()
cpu = dqn_player(config, device)
opponent = heuristic_player(cpu.opponent)

print("Iniciando")
if (device == "cuda"): cudnn.benchmark = True

# Training Episodes:
for i in range(training_episodes):
    print("training: ", i)
    game = neurohex_game(config.board_size, config.padding, device)
    turn = 0
    while (game.winner() == None):
        if (turn%2 == config.color):
            state = game.preprocess()
            action = cpu.select_valid_action(state, game.legal_actions())
            game.play(cpu.action_to_index(action))
            next_state = game.preprocess()

        else:
            opponent.eletric_resistence_play(game)
            if (game.winner() == None and turn != 0):
                cpu.play_reward(action, state, next_state)
        turn += 1
    
    print("winner: ", game.winner(), "\n")
    if (game.winner() == config.color):
        cpu.win_reward(action, state, next_state)
    else:
        cpu.lose_reward(action, state, next_state)

wins = 0
momentum = 0
max_momentum = 0
plot = []
for i in range(test_episodes):
    print("testing: ", i)
    game = neurohex_game(config.board_size, config.padding, device)
    turn = 0
    while (game.winner() == None):
        if (turn%2 == config.color):
            state = game.preprocess()
            action = cpu.select_valid_action(state, game.legal_actions())
            game.play(cpu.action_to_index(action))
            next_state = game.preprocess()

        else:
            opponent.random_play(game)
            if (game.winner() == None and turn != 0):
                cpu.play_reward(action, state, next_state)
        turn += 1
    
    print("winner: ", game.winner(), "\n")
    print(game)
    if (game.winner() == config.color):
        cpu.win_reward(action, state, next_state)
        wins+=1
        momentum +=1
        if (momentum > max_momentum):
            max_momentum = momentum
    else:
        cpu.lose_reward(action, state, next_state)
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