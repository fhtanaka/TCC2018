import sys
sys.path.append('../')
from network import *
from player_agents import *
from game_model import *
from config import *
from tqdm import tqdm
import matplotlib.pyplot as plt


def training_cpu(player_model, opponent, train, batch_wins_size, min_wins):
    wins = np.zeros(batch_wins_size)
    index_wins = 0
    i = 0

    while np.sum(wins) <= min_wins:
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
                if (game.winner() == None and turn != 0 and train == "mirror"):
                    cpu.play_reward(action, state, torch.tensor(game.super_board))
            turn += 1
        
        if (game.winner() == player_model.color):
            cpu.win_reward(action, state, game.zero_board())
            cpu.win_reward(action, state, game.zero_board())
            if (train == "mirror"):
                cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
                cpu.lose_reward_turn_influenced(op_action, op_state, game.zero_board(), turn)
            wins[index_wins] = 1

        else:
            cpu.lose_reward_turn_influenced(action, state, game.zero_board(), turn)
            cpu.lose_reward_turn_influenced(action, state, game.zero_board(), turn)
            if (train == "mirror"):
                cpu.win_reward(op_action, op_state, game.zero_board())
                cpu.win_reward(op_action, op_state, game.zero_board())

        if (i%cpu.policy_net_update == 0 and i > 0):
            loss = cpu.optimize_policy_net()
            #     plot.append(loss.item())

        if (i%cpu.target_net_update == 0 and i > 0):
            cpu.optimize_target_net()

        i+=1
        index_wins = (index_wins + 1)%batch_wins_size
    print("\t" + str(i))
    return i


print(device, "\n")
if (torch.cuda.is_available()):
    torch.backends.cudnn.benchmark = True

number_of_tests = 10
training=["mirrored", "simple"]
opponent=["random"]
batch_wins_size=100
min_wins=90
media = 0
for opp in opponent:
    print(opp)
    for train in training:
        print("\t" + train)
        for i in range(0, number_of_tests):
            cpu = dqn_player(config(), device)
            adv = heuristic_player(cpu.opponent, opp, chance=0.8)
            testes = training_cpu(cpu, adv, train, batch_wins_size, min_wins)
            media += testes
        print(media/number_of_tests)
        media = 0
        print()
    print()

