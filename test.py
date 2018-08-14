from config import config
from network import *
from player_agents import *
from hex_client import *
import matplotlib
import matplotlib.pyplot as plt

print("Iniciando")

training_episodes = 1000
test_episodes = 100
board_size = 5
wins = 0
momentum = 0
max_momentum = 0

config = config()
cpu = cpu_player("white", board_size, config)

for i in range(training_episodes):
    print("training: ", i)
    game = gamestate(board_size)
    turn = 0
    while (game.moves() != [] and game.winner() == 0):
        if (turn%2 == 0):
            valid = False
            while not valid:
                state = cpu.preprocess(game)
                action = cpu.select_action(game)
                valid, end, reward = cpu.reward(game, action, 1)
                next_state = cpu.preprocess(game) if valid else None
                cpu.memory.push(state, action, next_state, reward)
                cpu.optimize_model()
        else:
            game.random_play()
        turn += 1

plot = []
for i in range (test_episodes):
    print("test: ", i)
    game = gamestate(board_size)
    turn = 0
    while (game.moves() != [] and game.winner() == 0):
        if (turn%2 == 0):
            valid = False
            while not valid:
                state = cpu.preprocess(game)
                action = cpu.select_action(game, optimal=True)
                valid, end, reward = cpu.reward(game, action, 1)
                next_state = cpu.preprocess(game) if valid else None
                cpu.memory.push(state, action, next_state, reward)
                cpu.optimize_model()
        else:
            game.random_play()
        turn += 1
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
plt.ylim(0, test_episodes)
plt.plot(plot)
plt.show()



# tensor([-0.2889, -0.0611], grad_fn=<SelectBackward>)
