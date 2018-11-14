import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def training(player_model, num_episodes, opponent_method, filename=False, boards_to_print=-1):
    opponent = heuristic_player(cpu.opponent, opponent_method, chance=0.8)
    wins = 0
    momentum = 0
    max_momentum = 0
    games_string=""
    losses_plt = []
    wins_each_100 = 0
    wins_plt = []

    print("Beginning", opponent_method, " training of ", num_episodes, " episodes")
    for i in tqdm(range(num_episodes), desc=opponent_method+" training (" + str(num_episodes)+ ")"):
        game = hex_game(player_model.board_size, player_model.padding, device)
        game.change_color_print()
        turn = 0
        while (game.winner() == None):
            
            if (turn%2 == player_model.color):
                state = torch.tensor(game.super_board)
                action = cpu.select_valid_action(game)
                game.play(game.action_to_index(action))
                if (game.winner() == None and turn != 0):
                    cpu.play_reward(op_action, op_state, torch.tensor(game.mirror_board()))


            else:
                op_state = torch.tensor(game.mirror_board())
                op_action = torch.tensor([[opponent.play(game)]], device=device, dtype=torch.long)

                if (game.winner() == None and turn != 0):
                    cpu.play_reward(action, state, torch.tensor(game.super_board))
            turn += 1
        
        if (game.winner() == player_model.color):
            cpu.wins += 1
            cpu.win_reward(action, state, game.zero_board())
            cpu.win_reward(action, state, game.zero_board())
            cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
            cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
            wins+=1
            momentum +=1
            wins_each_100+=1
            if (momentum > max_momentum):
                max_momentum = momentum
        else:
            cpu.lose_reward_turn_influenced(action, state, game.zero_board(), turn)
            cpu.lose_reward_turn_influenced(action, state, game.zero_board(), turn)
            cpu.win_reward(op_action, op_state, game.zero_board())
            cpu.win_reward(op_action, op_state, game.zero_board())
            momentum = 0

        if (i%cpu.policy_net_update == 0 and i > 0):
            loss = cpu.optimize_policy_net()
            if (loss is not None):
                losses_plt.append(loss.item())
            #     plot.append(loss.item())

        if (i%cpu.target_net_update == 0 and i > 0):
            cpu.optimize_target_net()

        if (boards_to_print != -1 and i%boards_to_print == 0):
            games_string += "\nGame " + str(i) + ":\n"
            games_string += "Winner: " + str(game.winner())
            games_string += game.__str__() + "\n"

        if (i%100==0 and i > 0):
            wins_plt.append(wins_each_100)
            wins_each_100 = 0

    print("Number of games: " + str(num_episodes))
    print("Agent: Mirror")
    print("Opponent: " + str(opponent_method))
    if (opponent_method == "mixed"):
        print("chance of eletric: 0.8")
    print("Number of wins: " + str(wins))
    print("Win percentage: " + str(wins/num_episodes))
    print("Max consecutives wins: " + str(max_momentum) + "\n")
    if (filename != False):
        file = open(filename, "w")
        file.write("Number of games: " + str(num_episodes))
        file.write("Agent: Mirror")
        file.write("Opponent: " + str(opponent_method))
        if (opponent_method == "mixed"):
            file.write("chance of eletric: 0.8")
        file.write("\nNumber of wins: " + str(wins))
        file.write("\nWin percentage: " + str(wins/num_episodes))
        file.write("\nMax consecutives wins: " + str(max_momentum))
        file.write("\n\n")
        file.write(games_string)
        file.write("\n\n")
        file.close

        plt.subplot(211)
        plt.title('Wins each 100 games')
        print(wins_plt)
        plt.plot(wins_plt)
        plt.subplot(212)
        plt.title('Loss function value')
        plt.plot(losses_plt)

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(filename + '_image.png')
        plt.clf()


color = white
save = True

print(device, "\n")
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

cpu = dqn_player(config(white), device)
for ep in training_regime:
    training(cpu, *ep)
    
if (save):
    torch.save(cpu.policy_net, 'white_train_7.pt')

