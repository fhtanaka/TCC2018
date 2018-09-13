from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
# import matplotlib
# import matplotlib.pyplot as plt


device = "cpu"
print(device)
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpuW = dqn_player(config(white), device)
cpuB = dqn_player(config(black), device)

num_episodes = 100000
print_to_file = True
boards_to_print = 100

winsW = winsB = 0
momentumW = momentumB = 0
max_momentumW = max_momentumB = 0
games_string = ""
    

print("Beginning versus training of ", num_episodes, " episodes")
for i in tqdm(range(num_episodes), desc="versus training (" + str(num_episodes)+ ")"):
    
    game = hex_game(cpuW.board_size, cpuW.padding, device)
    turn = 0

    while (game.winner() == None):

        if (turn%2 == cpuW.color):
            stateW = game.preprocess()
            actionW = cpuW.select_valid_action(stateW, game.legal_actions())
            game.play(cpuW.action_to_index(actionW))
            next_stateW = game.preprocess()
            if (game.winner() == None and turn > 1):
                cpuB.play_reward(actionB, stateB, next_stateB)

        else:
            stateB = game.preprocess()
            actionB = cpuB.select_valid_action(stateB, game.legal_actions())
            game.play(cpuB.action_to_index(actionB))
            next_stateB = game.preprocess()
            if (game.winner() == None and turn > 1):
                cpuW.play_reward(actionW, stateW, next_stateW)

        turn += 1
    
    if (game.winner() == cpuW.color):
        cpuW.win_reward(actionW, stateW, next_stateW)
        cpuB.lose_reward_turn_influenced(actionB, stateB, next_stateB, turn)
        winsW+=1
        momentumW +=1
        momentumB = 0
        if (momentumW > max_momentumW):
            max_momentumW = momentumW
    else:
        cpuB.win_reward(actionB, stateB, next_stateB)
        cpuW.lose_reward_turn_influenced(actionW, stateW, next_stateW, turn)
        winsB+=1
        momentumB +=1
        momentumW = 0
        if (momentumB > max_momentumB):
            max_momentumB = momentumB
    # plot.append(wins)

    if (i%cpuW.target_update == 0):
        cpuW.optimize_target_net()
        cpuB.optimize_target_net()

    if (print_to_file == True and i%boards_to_print == 1):
        games_string += "\nGame " + str(i) + ":\n"
        games_string += "Winner: " + str(game.winner())
        games_string += game.str_colorless() + "\n"

print("White: ")
print("\tWin percentage: " + str(winsW/num_episodes))
print("\tMax consecutives wins: " + str(max_momentumW) + "\n")

print("Black: ")
print("\tWin percentage: " + str(winsB/num_episodes))
print("\tMax consecutives wins: " + str(max_momentumB) + "\n")

if (print_to_file):
    file = open("VS_test.txt", "w")

    file.write("White: ")
    file.write("\tWin percentage: " + str(winsW/num_episodes))
    file.write("\tMax consecutives wins: " + str(max_momentumW) + "\n")

    file.write("Black: ")
    file.write("\tWin percentage: " + str(winsB/num_episodes))
    file.write("\tMax consecutives wins: " + str(max_momentumB) + "\n")

    file.write("\n\n")
    file.write(games_string)
    # file.write(str(plot))
    file.write("\n\n")
    file.close

torch.save(cpuW.policy_net, 'white_train.pt')
torch.save(cpuB.policy_net, 'black_train.pt')