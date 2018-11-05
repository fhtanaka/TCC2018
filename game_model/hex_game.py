import torch
import numpy as np
import random

white = 0
black = 1
west = 2
east = 3
north = 4
south = 5
neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))

def other(color):
    if (color==black):
        return white
    else:
        return black
'''
This game representation uses 6 channels as follows:
    white stone present
    black stone present
    white stone group connected to left edge
    white stone group connected to right edge
    black stone group connected to top edge
    black stone group connected to bottom edge
'''
class hex_game:

    def __init__(self, size, padding, device="cpu", board=torch.empty(0)):
        self.device=device
        self.size = size
        self.padding = padding
        self.input_size = size+2*padding
        self.input_shape = (6,self.input_size,self.input_size)
        self.to_play=white
        self.print_colored = True

        if (len(board) != 0):
            self.super_board = torch.tensor(board)
            self.torch_board = self.super_board[0]
        else:
            self.super_board = torch.zeros((1, *self.input_shape), dtype=torch.float, device=self.device)
            self.torch_board = self.super_board[0]
            self.torch_board[white, 0:padding, :] = 1
            self.torch_board[white, self.input_size-padding:, :] = 1
            self.torch_board[west, 0:padding, :] = 1
            self.torch_board[east, self.input_size-padding:, :] = 1
            self.torch_board[black, :, 0:padding] = 1
            self.torch_board[black, :, self.input_size-padding:] = 1
            self.torch_board[north, :, 0:padding] = 1
            self.torch_board[south, :, self.input_size-padding:] = 1

        self.np_board = self.torch_board.cpu().numpy()

    '''
    There are 3 types of notation in the program:
    1) Notation: This is the "human reading" of the play (Ex: a2)
    2) Index: This is the coordinate on the matrix of the play (Ex: (1,2))
    3) Action: This is the action related to a specific play (Ex: 1 means (1,1), 2 means (1,2)...)
    '''

    def index_to_notation(self, index):
        return chr(ord('a')+index[1]-self.padding)+str(index[0]-self.padding+1)

    def index_to_action(self, index):
        return (index[0]-self.padding)*self.size+(index[1]-self.padding)

    def notation_to_index(self, notation):
        x = int(notation[1:])-1+self.padding
        y = ord(notation[0].lower())-ord('a')+self.padding
        return (x,y)

    def notation_to_action(self, notation):
        return self.index_to_action(self.notation_to_index(notation))

    def action_to_index(self, action):
        i = int(action/self.size)+self.padding
        j = int(action%self.size)+self.padding
        return (i,j)

    def action_to_notation(self, action):
        return self.index_to_notation(self.action_to_index(action))

    # Changes any type of notation to index
    def change_move_to_index(self, move):
        tipo = type(move)
        if (tipo == int):
            return self.action_to_index(move)
        elif (tipo == str):
            return self.notation_to_index(move)
        else:
            return move

    #cell of the mirrored move
    def cell_m(self, cell):
        return (cell[1],cell[0])

    # Return list of neighbors of the passed cell.
    def neighbors(self, cell):
        """
        """
        x = cell[0]
        y = cell[1]
        return [(n[0]+x , n[1]+y) for n in neighbor_patterns\
            if (0<=n[0]+x and n[0]+x<self.input_size and 0<=n[1]+y and n[1]+y<self.input_size)]

    # def mirror_board(self):
    #     m_board = np.zeros(self.np_board.shape, dtype=float)
    #     m_board[white]=np.transpose(self.np_board[black])
    #     m_board[black]=np.transpose(self.np_board[white])
    #     m_board[north]=np.transpose(self.np_board[west])
    #     m_board[east] =np.transpose(self.np_board[south])
    #     m_board[south]=np.transpose(self.np_board[east])
    #     m_board[west] =np.transpose(self.np_board[north])

    #     return torch.from_numpy(m_board).to(self.device)

    def mirror_board(self):
        m_board = torch.zeros((1, *self.input_shape), dtype=torch.float, device=self.device)
        m_board[0][white] = torch.transpose(self.torch_board[black], 0, 1)
        m_board[0][black] = torch.transpose(self.torch_board[white], 0, 1)
        m_board[0][north] = torch.transpose(self.torch_board[west], 0, 1)
        m_board[0][east]  = torch.transpose(self.torch_board[south], 0, 1)
        m_board[0][south] = torch.transpose(self.torch_board[east], 0, 1)
        m_board[0][west]  = torch.transpose(self.torch_board[north], 0, 1)
        return m_board


    def winner(self):
        if(self.np_board[east,0,0] and self.np_board[west,0,0]):
            return white
        elif(self.np_board[north,0,0] and self.np_board[south,0,0]):
            return black
        return None

    #return 1 if color wins, -1 if it loses and 0 otherwise
    def is_winner(self, color):
        winner = self.winner()
        if (color==winner):
            return 1
        elif (other(color)==winner):
            return -1
        else:
            return 0

    def flood_fill(self, cell, color, edge):
        self.np_board[edge, cell[0], cell[1]] = 1
        self.torch_board[edge, cell[0], cell[1]] = 1
        for n in self.neighbors(cell):
            if(self.np_board[color, n[0], n[1]] and not self.np_board[edge, n[0], n[1]]):
                self.flood_fill(n, color, edge)

    def play_cell(self, cell, color):
        edge1_connection = False
        edge2_connection = False
        cell = self.change_move_to_index(cell)
        if self.np_board[other(color), cell[0], cell[1]] == 1 or self.np_board[color, cell[0], cell[1]] == 1:
            raise ValueError("Cell occupied")
        
        #Fill the classe of the color
        self.torch_board[color, cell[0], cell[1]] = 1
        self.np_board[color, cell[0], cell[1]] = 1
        
        # Fill the connected classes
        if(color == white):
            edge1 = east
            edge2 = west
        else:
            edge1 = north
            edge2 = south
        for n in self.neighbors(cell):
            if(self.np_board[edge1, n[0], n[1]] and self.np_board[color, n[0], n[1]]):
                edge1_connection = True
            if(self.np_board[edge2, n[0], n[1]] and self.np_board[color, n[0], n[1]]):
                edge2_connection = True
        if(edge1_connection):
            self.flood_fill(cell, color, edge1)
        if(edge2_connection):
            self.flood_fill(cell, color, edge2)

    def play(self, cell):
        if (self.to_play == white):
            self.play_cell(cell, white)
            self.to_play=black
        else:
            self.play_cell(cell, black)
            self.to_play=white

    # Returns the indexes that are possible to play (Where there is no piece)
    # def legal_indexes(self):
    #     white_plays = (self.torch_board[white]==0).nonzero().numpy()
    #     black_plays = (self.torch_board[black]==0).nonzero().numpy()
    #     possible_plays = np.array([x for x in set(tuple(x) for x in black_plays) & set(tuple(x) for x in white_plays)])
    #     return possible_plays

    def legal_indexes(self):
        white_plays=np.argwhere(self.np_board[white]==False)
        black_plays=np.argwhere(self.np_board[black]==False)
        possible_plays = np.array([x for x in set(tuple(x) for x in black_plays) & set(tuple(x) for x in white_plays)])
        return possible_plays

    def legal_actions(self):
        return list(map(self.index_to_action, self.legal_indexes()))

    def random_play(self, color=None):
        possible_plays = self.legal_indexes()
        if possible_plays.shape[0] != 0:
            i, j = random.choice(possible_plays)
            if (color != None):
                self.play_cell((i,j), color)
            else:
                self.play((i,j))
            return (i,j)
        else:
            raise ValueError("No possible plays")

    def is_cell_empty(self, cell):
        if (self.np_board[white, cell[0], cell[1]]==False and self.np_board[black, cell[0], cell[1]]==False):
            return True
        else:
            return False

    def change_color_print(self):
        self.print_colored = not self.print_colored

    def zero_board(self):
        return torch.zeros(self.super_board.size(), device=self.device, dtype=torch.float)

    def __str__(self):
        """
        Print an ascii representation of the input.
        """
        w = '#'
        b = '@'
        empty = '.'
        if (self.print_colored):
            end_color = '\033[0m'
            edge1_color = '\033[31m'
            edge2_color = '\033[32m'
            both_color =  '\033[33m'
        else:
            end_color = ''
            edge1_color = ''
            edge2_color = ''
            both_color =  ''
        invalid = '*'
        ret = '\n'
        coord_size = len(str(self.size))
        offset = 1
        ret+=' '*(offset+2)
        for x in range(self.input_size):
            if(x<self.padding or x>=self.size+self.padding):
                ret+=' '*(offset*2+1)
            else:
                ret+=chr(ord('A')+(x-self.padding))+' '*offset*2
        ret+='\n'
        for x in range(self.input_size):
            if(x<self.padding or x>=self.size+self.padding):
                ret+=' '*(offset*2+coord_size)
            else:
                ret+=str(x+1-self.padding)+' '*(offset*2+coord_size-len(str(x+1-self.padding)))
            for y in range(self.input_size):
                if(self.np_board[white, x, y] == 1):
                    if(self.np_board[west, x, y] == 1 and self.np_board[east, x, y]):
                        ret+=both_color
                    elif(self.np_board[west, x, y]):
                        ret+=edge1_color
                    elif(self.np_board[east, x, y]):
                        ret+=edge2_color
                    if(self.np_board[black, x, y] == 1):
                        ret+=invalid
                    else:
                        ret+=w
                    ret+=end_color
                elif(self.np_board[black, x, y] == 1):
                    if(self.np_board[north, x, y] == 1 and self.np_board[south, x, y]):
                        ret+=both_color
                    elif(self.np_board[north, x, y]):
                        ret+=edge1_color
                    elif(self.np_board[south, x, y]):
                        ret+=edge2_color
                    ret+=b
                    ret+=end_color
                else:
                    ret+=empty
                ret+=' '*offset*2
            ret+="\n"+' '*offset*(x+1)
        ret+=' '*(offset*2+1)+(' '*offset*2)*self.input_size

        return ret