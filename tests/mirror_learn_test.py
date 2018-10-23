import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def training(player_model, num_episodes, opponent_method, filename=False, boards_to_print=-1):
    opponent = heuristic_player(cpu.opponent, 0.5)
    if (opponent_method == "random"):
        opponent_play = opponent.random_play
    elif (opponent_method == "eletric"):
        opponent_play = opponent.eletric_resistence_play
    elif (opponent_method == "mixed"):
        opponent_play = opponent.mixed_play

    wins = 0
    momentum = 0
    max_momentum = 0
    games_string=""
    plot=[]
        

    print("Beginning", opponent_method, " training of ", num_episodes, " episodes")
    for i in tqdm(range(num_episodes), desc=opponent_method+" training (" + str(num_episodes)+ ")"):
        game = hex_game(player_model.board_size, player_model.padding, device)
        turn = 0
        if (i%boards_to_print == 1):
            games_string += "\nGame " + str(i) + ":\n"
        while (game.winner() == None):
            if (turn%2 == player_model.color):
                state = game.preprocess()
                action = cpu.select_valid_action(state, game.legal_actions())
                game.play(cpu.action_to_index(action))
                next_state = game.preprocess()

                if (game.winner() == None and turn != 0):
                    cpu.play_reward(op_action, op_state, op_next_state)

            else:
                op_state = game.mirrored_preprocess()
                op_action = opponent_play(game)
                op_action = torch.tensor([[op_action]], device=device, dtype=torch.long)
                op_next_state = game.mirrored_preprocess()

                if (game.winner() == None and turn != 0):
                    cpu.play_reward(action, state, next_state)

            if (i%boards_to_print == 1):
                games_string += "\n" + game.str_colorless() + "\n"
            turn += 1
        
        if (game.winner() == player_model.color):
            cpu.win_reward(action, state, next_state)
            cpu.lose_reward(op_action, op_state, op_next_state)
            wins+=1
            momentum +=1
            if (momentum > max_momentum):
                max_momentum = momentum
        else:
            cpu.win_reward(op_action, op_state, op_state)
            cpu.lose_reward(action, state, next_state)
            momentum = 0
        plot.append(wins)

        if (i%cpu.target_update == 0):
            cpu.optimize_target_net()

        if (i%boards_to_print == 1):
            games_string += "Winner: " + str(game.winner())
            games_string += "\n" + "!"*50 + "\n"
            # games_string += "\nGame " + str(i) + ":\n"
            # games_string += "Winner: " + str(game.winner())
            # games_string += game.str_colorless() + "\n"

        cpu.optimize_model()


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

    plt.ylim(0, num_episodes)
    plt.plot(plot)
    plt.show()


config = config(white)

print(device)
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = dqn_player(config, device)
for ep in training_regime:
    training(cpu, ep[0], ep[1], ep[2], ep[3])

# torch.save(cpu.policy_net, 'white_train.pt')