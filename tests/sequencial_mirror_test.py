import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt

def training (player_model, num_episodes, opponent_model, number, file=False, filename=False, boards_to_print=-1):
    opponent = opponent_model
    wins = 0
    momentum = 0
    max_momentum = 0
    games_string=""
    losses_plt = []
    wins_each_200 = 0
    wins_plt = []
    # plot=[]

    print("Beginning training " + str(number) + " of ", num_episodes, " episodes")
    for i in tqdm(range(num_episodes)):
        game = hex_game(player_model.board_size, player_model.padding, device)
        game.change_color_print()
        turn = 0
        while (game.winner() == None):
            
            if (turn%2 == player_model.color):
                state = torch.tensor(game.super_board)
                action = cpu.select_valid_action(game)
                game.play(game.action_to_index(action))
                next_state = torch.tensor(game.super_board)
                if (game.winner() == None and turn != 0):
                    cpu.play_reward(op_action, op_state, torch.tensor(game.mirror_board()))


            else:
                op_state = torch.tensor(game.mirror_board())
                op_action = torch.tensor([[opponent.play(game)]], device=device, dtype=torch.long)
                op_next_state = torch.tensor(game.mirror_board())

                if (game.winner() == None and turn != 0):
                    cpu.play_reward(action, state, torch.tensor(game.super_board))
            turn += 1
        
        if (game.winner() == player_model.color):
            # print("Ganhou!!! (", wins+1, ")")
            cpu.wins += 1
            cpu.win_reward(action, state, game.zero_board())
            cpu.win_reward(action, state, game.zero_board())
            cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
            cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
            wins+=1
            wins_each_200+=1
            momentum +=1
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

        if (i%cpu.target_net_update == 0 and i > 0):
            cpu.optimize_target_net()

        if (boards_to_print != -1 and i%boards_to_print == 0):
            games_string += "\nGame " + str(i) + ":\n"
            games_string += "Winner: " + str(game.winner())
            games_string += game.__str__() + "\n"

        if (i%200==0 and i > 0):
            wins_plt.append(wins_each_200)
            wins_each_200 = 0


    if (file != False):
        file.write("Number of games: " + str(num_episodes) + "\n")
        file.write("Agent: Mirror 0.8\n")
        file.write("\nNumber of wins: " + str(wins))
        file.write("\nWin percentage: " + str(wins/num_episodes))
        file.write("\nMax consecutives wins: " + str(max_momentum))
        file.write("\n\n")
        file.write(games_string)
        # file.write(str(plot))
        file.write("\n\n")

        plt.subplot(211)
        plt.title('Wins each 200 games')
        plt.plot(wins_plt)
        plt.subplot(212)
        plt.title('Loss function value')
        plt.plot(losses_plt)

        plt.subplots_adjust(hspace=0.5)
        plt.savefig(filename + '_image.png')
        plt.clf()




if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True


learning_rates = [0.01]
gamma = [0.8]
eps_end = [0.1]
layers = [([48, 384], [2, 2])]
boards = [9]

# learning_rates = [0.1, 0.01]
# gamma = [0.9, 0.8, 0.99]
# eps_end = [0.1, 0.25, 0.4]
# layers = [([48, 384], [2, 2])]

# learning_rates = [0.01]
# gamma = [0.8]
# eps_end = [0.1]
# layers = [([48, 384], [2, 2])]

training_episodes=100000
b_print = 100

training_number = 0
for lr in learning_rates:
    for gm in gamma:
        for end in eps_end:
            for layer in layers:
                for board in boards:
                    configuration = config(
                        board_size=board,
                        lr=lr, 
                        gamma=gm, 
                        eps_end=end, 
                        conv_layers=layer[0],
                        kernel=layer[1]
                        )
                    cpu = dqn_player(configuration, device)
                    opp = heuristic_player(cpu.opponent, "mixed", chance=0.8)

                    filename="results_boards/"+"3_board-"+str(board) + "_lr-" + str(lr)+ "_gm-" + str(gm) + "_end-" + str(end) + "_layer-" + str(len(layer[0]))
                    # filename="results_boards/"+str(training_number)+"_board-"+str(board) + "_lr-" + str(lr)+ "_gm-" + str(gm) + "_end-" + str(end) + "_layer-" + str(len(layer[0]))

                    print("\n"+filename)
                    file = open(filename, "w")

                    file.write("board-" + str(board) + "\n")
                    file.write("lr-" + str(lr) + "\n")
                    file.write("gamma-" + str(gm) + "\n")
                    file.write("end-" + str(end) + "\n")
                    file.write("layers-" + str(layer) + "\n\n")

                    training(cpu, training_episodes, opp, training_number, file=file, filename=filename, boards_to_print=b_print)

                    file.close()
                    training_number+=1


