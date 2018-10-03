import numpy as np
import torch
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

    def __init__(self, size, padding, device, board=np.empty(0)):
        self.device=device
        self.size = size
        self.padding = padding
        self.input_size = size+2*padding
        self.input_shape = (6,self.input_size,self.input_size)
        self.to_play=white

        if (board.any()):
            self.board=board
            return

        even = 1 - size%2
        self.board = np.zeros(self.input_shape, dtype=bool)
        self.board[white, 0:padding, :] = 1
        self.board[white, self.input_size-padding+even:, :] = 1
        self.board[west, 0:padding, :] = 1
        self.board[east, self.input_size-padding+even:, :] = 1
        self.board[black, :, 0:padding] = 1
        self.board[black, :, self.input_size-padding+even:] = 1
        self.board[north, :, 0:padding] = 1
        self.board[south, :, self.input_size-padding+even:] = 1

    # convert a move in human notation like a1 to the index in the self.board
    # e.g. in a 5x5 board with padding lenght of 1 'a1'=(1,1)
    def notation_to_index(self, move):
        x = ord(move[0].lower())-ord('a')+self.padding
        y = int(move[1:])-1+self.padding
        return (x,y)

    # does the opposte of the above function
    def index_to_notation(self, cell):
        return chr(ord('a')+cell[0]-self.padding)+str(cell[1]-self.padding+1)

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

    def mirror_board(self):
        m_board = np.zeros(self.board.shape, dtype=bool)
        m_board[white]=np.transpose(self.board[black])
        m_board[black]=np.transpose(self.board[white])
        m_board[north]=np.transpose(self.board[west])
        m_board[east] =np.transpose(self.board[south])
        m_board[south]=np.transpose(self.board[east])
        m_board[west] =np.transpose(self.board[north])
        return m_board

    def flip_board(self):
        f_board = np.zeros(self.board.shape, dtype=bool)
        f_board[white] = np.rot90(self.board[white],2)
        f_board[black] = np.rot90(self.board[black],2)
        f_board[north] = np.rot90(self.board[south],2)
        f_board[east]  = np.rot90(self.board[west],2)
        f_board[south] = np.rot90(self.board[north],2)
        f_board[west]  = np.rot90(self.board[east],2)
        return f_board

    def winner(self):
        if(self.board[east,0,0] and self.board[west,0,0]):
            return white
        elif(self.board[north,0,0] and self.board[south,0,0]):
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
        self.board[edge, cell[0], cell[1]] = 1
        for n in self.neighbors(cell):
            if(self.board[color, n[0], n[1]] and not self.board[edge, n[0], n[1]]):
                self.flood_fill(n, color, edge)

    def play_cell(self, cell, color):
        edge1_connection = False
        edge2_connection = False
        if self.board[other(color), cell[0], cell[1]] == 1 or self.board[color, cell[0], cell[1]] == 1:
            raise ValueError("Cell occupied")
        self.board[color, cell[0], cell[1]] = 1
        if(color == white):
            edge1 = east
            edge2 = west
        else:
            edge1 = north
            edge2 = south
        for n in self.neighbors(cell):
            if(self.board[edge1, n[0], n[1]] and self.board[color, n[0], n[1]]):
                edge1_connection = True
            if(self.board[edge2, n[0], n[1]] and self.board[color, n[0], n[1]]):
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

    '''
    This function adds one dimension to the board and converts it to torch.tensor
    It also converts bool to int in the process
    '''
    def preprocess(self):
        tensor=torch.from_numpy(self.board*1)
        tensor = torch.tensor(tensor, device=self.device, dtype=torch.float)
        
        new_board = torch.zeros((1, 6, self.input_size, self.input_size), device=self.device)
        new_board[0] = tensor
        return new_board

    def mirrored_preprocess(self):
        tensor=torch.from_numpy(self.mirror_board()*1)
        tensor = torch.tensor(tensor, device=self.device, dtype=torch.float)
        new_board = torch.zeros((1, 6, self.input_size, self.input_size), device=self.device)
        new_board[0] = tensor
        return new_board

    def legal_actions(self):
        white_plays=np.argwhere(self.board[white]==False)
        black_plays=np.argwhere(self.board[black]==False)
        possible_plays = np.array([x for x in set(tuple(x) for x in black_plays) & set(tuple(x) for x in white_plays)])
        return possible_plays

    def action_to_tuple(self, action):
        i = int(action/self.size)+self.padding
        j = int(action%self.size)+self.padding
        return (i,j)

    def tuple_to_action(self, tuple):
        return (tuple[0]-self.padding)*self.size+(tuple[1]-self.padding)

    def random_play(self, color=None):
        #np.array([x for x in set(tuple(x) for x in black) & set(tuple(x) for x in white)])
        possible_plays = self.legal_actions()
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
        if (self.board[white, cell[0], cell[1]]==False and self.board[black, cell[0], cell[1]]==False):
            return True
        else:
            return False


    def str_colorless(self):
        """
        Print an ascii representation of the input.
        """
        w = '#'
        b = '@'
        empty = '.'
        end_color = ""
        edge1_color = ""
        edge2_color = ""
        both_color =  ""
        invalid = '*'
        ret = '\n'
        coord_size = len(str(self.size))
        offset = 1
        ret+=' '*(offset+2)
        for x in range(1, self.input_size-1):
            if(x<self.padding or x>=self.size+self.padding):
                ret+=' '*(offset*2+1)
            else:
                ret+=chr(ord('A')+(x-self.padding))+' '*offset*2
        ret+='\n'
        for y in range(1, self.input_size-1):
            if(y<self.padding or y>=self.size+self.padding):
                ret+=' '*(offset*2+coord_size)
            else:
                ret+=str(y+1-self.padding)+' '*(offset*2+coord_size-len(str(y+1-self.padding)))
            for x in range(1, self.input_size-1):
                if(self.board[white, x, y] == 1):
                    if(self.board[west, x, y] == 1 and self.board[east, x, y]):
                        ret+=both_color
                    elif(self.board[west, x,y]):
                        ret+=edge1_color
                    elif(self.board[east, x, y]):
                        ret+=edge2_color
                    if(self.board[black, x, y] == 1):
                        ret+=invalid
                    else:
                        ret+=w
                    ret+=end_color
                elif(self.board[black, x, y] == 1):
                    if(self.board[north, x, y] == 1 and self.board[south, x, y]):
                        ret+=both_color
                    elif(self.board[north, x,y]):
                        ret+=edge1_color
                    elif(self.board[south, x, y]):
                        ret+=edge2_color
                    ret+=b
                    ret+=end_color
                else:
                    ret+=empty
                ret+=' '*offset*2
            ret+="\n"+' '*offset*(y+1)
        ret+=' '*(offset*2+1)+(' '*offset*2)*self.input_size

        return ret  

    def __str__(self):
        """
        Print an ascii representation of the input.
        """
        w = '#'
        b = '@'
        empty = '.'
        end_color = '\033[0m'
        edge1_color = '\033[31m'
        edge2_color = '\033[32m'
        both_color =  '\033[33m'
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
        for y in range(self.input_size):
            if(y<self.padding or y>=self.size+self.padding):
                ret+=' '*(offset*2+coord_size)
            else:
                ret+=str(y+1-self.padding)+' '*(offset*2+coord_size-len(str(y+1-self.padding)))
            for x in range(self.input_size):
                if(self.board[white, x, y] == 1):
                    if(self.board[west, x, y] == 1 and self.board[east, x, y]):
                        ret+=both_color
                    elif(self.board[west, x,y]):
                        ret+=edge1_color
                    elif(self.board[east, x, y]):
                        ret+=edge2_color
                    if(self.board[black, x, y] == 1):
                        ret+=invalid
                    else:
                        ret+=w
                    ret+=end_color
                elif(self.board[black, x, y] == 1):
                    if(self.board[north, x, y] == 1 and self.board[south, x, y]):
                        ret+=both_color
                    elif(self.board[north, x,y]):
                        ret+=edge1_color
                    elif(self.board[south, x, y]):
                        ret+=edge2_color
                    ret+=b
                    ret+=end_color
                else:
                    ret+=empty
                ret+=' '*offset*2
            ret+="\n"+' '*offset*(y+1)
        ret+=' '*(offset*2+1)+(' '*offset*2)*self.input_size

        return ret