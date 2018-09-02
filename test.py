from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
# import matplotlib
# import matplotlib.pyplot as plt

def training(player_model, num_episodes, opponent_method, filename=False, boards_to_print=-1):
    opponent = heuristic_player(cpu.opponent)
    if (opponent_method == "random"):
        opponent_play = opponent.random_play
    elif (opponent_method == "eletric"):
        opponent_play = opponent.eletric_resistence_play

    wins = 0
    momentum = 0
    max_momentum = 0
    games_string=""
    plot=[]
        

    print("Beggining", opponent_method, " training of ", num_episodes, " episodes")
    for i in tqdm(range(num_episodes), desc=opponent_method+" training (" + str(num_episodes)+ ")"):
        game = hex_game(player_model.board_size, player_model.padding, device)
        turn = 0
        while (game.winner() == None):
            if (turn%2 == player_model.color):
                state = game.preprocess()
                action = cpu.select_valid_action(state, game.legal_actions())
                game.play(cpu.action_to_index(action))
                next_state = game.preprocess()

            else:
                opponent_play(game)
                if (game.winner() == None and turn != 0):
                    cpu.play_reward(action, state, next_state)
            turn += 1
        
        if (game.winner() == player_model.color):
            cpu.win_reward(action, state, next_state)
            wins+=1
            momentum +=1
            if (momentum > max_momentum):
                max_momentum = momentum
        else:
            cpu.lose_reward_turn_influenced(action, state, next_state, turn)
            momentum = 0
        plot.append(wins)

        if (i%boards_to_print ==1):
            games_string += "\nGame " + str(i) + ":\n"
            games_string += "Winner: " + str(game.winner())
            games_string += game.str_colorless() + "\n"


    print("Win percentage: " + str(wins/num_episodes))
    print("Max consecutives wins: " + str(max_momentum) + "\n")
    if (filename != False):
        file = open(filename, "w")
        file.write("\nWin percentage: " + str(wins/num_episodes))
        file.write("\nMax consecutives wins: " + str(max_momentum))
        file.write("\n\n")
        file.write(games_string)
        # file.write(str(plot))
        file.write("\n\n")
        file.close 


config = config()
print(device)
if (device == "cuda"): cudnn.benchmark = True

cpu = dqn_player(config, device)
for ep in training_regime:
    training(cpu, ep[0], ep[1], ep[2], ep[3])