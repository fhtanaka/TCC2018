import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm

def training(player_model, num_episodes, opponent_method, filename=False, boards_to_print=-1):
    opponent = heuristic_player(cpu.opponent, opponent_method, chance=0.8)
    wins = 0
    momentum = 0
    max_momentum = 0
    games_string=""
    plot=[]

    print("Beginning", opponent_method, " training of ", num_episodes, " episodes")
    for i in tqdm(range(num_episodes), desc=opponent_method+" training (" + str(num_episodes)+ ")"):
        game = hex_game(player_model.board_size, player_model.padding, device)
        game.change_color_print()
        turn = 0
        while (game.winner() == None):
            
            if (turn%2 == player_model.color):
                action = cpu.select_valid_action(game)
                game.play(game.action_to_index(action))
            else:
                opponent.play(game)
            turn += 1
        
        if (game.winner() == player_model.color):
            wins+=1
            momentum +=1
            if (momentum > max_momentum):
                max_momentum = momentum
        else:
            momentum = 0
        plot.append(wins)

        if (i%boards_to_print == 1):
            games_string += "\nGame " + str(i) + ":\n"
            games_string += "Winner: " + str(game.winner())
            games_string += game.__str__() + "\n"


    print("Number of wins: " + str(wins))
    print("Win percentage: " + str(wins/num_episodes))
    print("Max consecutives wins: " + str(max_momentum) + "\n")
    if (filename != False):
        file = open(filename, "w")
        file.write("\nNumber of wins: " + str(wins))        
        file.write("\nWin percentage: " + str(wins/num_episodes))
        file.write("\nMax consecutives wins: " + str(max_momentum))
        file.write("\n\n")
        file.write(games_string)
        # file.write(str(plot))
        file.write("\n\n")
        file.close 


color = white
save = False

print(device, "\n")
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = dqn_player(config(white), device, model="white_train.pt")
for ep in training_regime:
    training(cpu, *ep)
    
if (save):
    torch.save(cpu.policy_net, 'white_train.pt')

