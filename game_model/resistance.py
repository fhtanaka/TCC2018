import numpy as np
from game_model.hex_game import *
import sys

white = 0
black = 1
west = 2
east = 3
north = 4
south = 5
neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))


def get_empty(game):
	count = 0
	indices = []
	for x in range(game.input_size):
		for y in range(game.input_size):
			if(game.np_board[white,x,y] == 0 and game.np_board[black,x,y] == 0):
				count+=1
				indices.append((x,y))
	return count, indices


def fill_connect(game, cell, color, checked):
	checked[cell] = True
	connected = set()
	for n in game.neighbors(cell):
		if(not checked[n]):
			if(game.np_board[color,n[0], n[1]]):
				connected = connected | fill_connect(game, n, color, checked)
			elif(not game.np_board[other(color), n[0], n[1]]):
				connected.add(n)
	return connected


def get_connections(game, color, empty, checked):
	connections = {cell:set() for cell in empty}
	for cell in empty:
		for n in game.neighbors(cell):
			if(not checked[n]):
				if(game.np_board[color, n[0], n[1]]):
					connected_set = fill_connect(game, n, color, checked)
					for c1 in connected_set:
						for c2 in connected_set:
							connections[c1].add(c2)
							connections[c2].add(c1)
				elif(not game.np_board[other(color),n[0],n[1]]):
					connections[cell].add(n)
	return connections


def resistance(game, empty, color):
	"""
	Output a resistance heuristic score over all empty nodes:
		-Treat the west edge connected nodes as one source node with voltage 1
		-Treat east edge connected nodes as one destination node with voltage 0
		-Treat edges between empty nodes, or from source/dest to an empty node as resistors with conductance 1
		-Treat edges to white nodes (except source and dest) as perfect conductors
		-Treat edges to black nodes as perfect resistors
	"""
	if(game.winner()!=None):
		if game.winner() == color:
			return np.zeros((game.input_size, game.input_size)), float("inf")
		else:
			return np.zeros((game.input_size, game.input_size)), 0
	index_to_location = empty
	num_empty = len(empty)
	location_to_index = {index_to_location[i]:i for i in range(len(index_to_location))}

	#current through internal nodes except that from source and dest
	#(zero except for source and dest connected nodes)
	I = np.zeros(num_empty)

	#conductance matrix such that G*V = I
	G = np.zeros((num_empty, num_empty))

	checked = np.zeros((game.input_size, game.input_size), dtype=bool)
	source_connected = fill_connect(game, (0,0), color, checked)
	for n in source_connected:
		j = location_to_index[n]
		I[j] += 1
		G[j,j] += 1
		

	dest_connected = fill_connect(game, (game.input_size-1,game.input_size-1), color, checked)
	for n in dest_connected:
		j = location_to_index[n]
		G[j,j] +=1

	adjacency = get_connections(game, color, index_to_location, checked)
	for c1 in adjacency:
		j=location_to_index[c1]
		for c2 in adjacency[c1]:
			i=location_to_index[c2]
			G[i,j] -= 1
			G[i,i] += 1

	#voltage at each cell
	try:
		V = np.linalg.solve(G,I)
	#slightly hacky fix for rare case of isolated empty cells
	#happens rarely and fix should be fine but could improve
	except np.linalg.linalg.LinAlgError:
		try:
			V = np.linalg.lstsq(G,I, rcond=None)[0]
		except:
			V = np.linalg.lstsq(G,I)[0]

	V_board = np.zeros((game.input_size, game.input_size))
	for i in range(num_empty):
		V_board[index_to_location[i]] = V[i]

	#current passing through each cell
	Il = np.zeros((game.input_size, game.input_size))
	#conductance from source to dest
	C = 0

	for i in range(num_empty):
		if index_to_location[i] in source_connected:
			Il[index_to_location[i]] += abs(V[i] - 1)/2
		if index_to_location[i] in dest_connected:
			Il[index_to_location[i]] += abs(V[i])/2
		for j in range(num_empty):
			if(i!=j and G[i,j] != 0):
				Il[index_to_location[i]] += abs(G[i,j]*(V[i] - V[j]))/2
				if(index_to_location[i] in source_connected and
				 index_to_location[j] not in source_connected):
					C+=-G[i,j]*(V[i] - V[j])

	return Il, C

def score(game, color):
	current_credit = 2
	"""
	Score is an estimate of action value for each move, computed using the ratio
	of effective conductance for the current player and their opponent.
	The effective conductance is computed heuristically by taking the true 
	conductance and giving some credit for the current flowing through the cell 
	to be played for both the player and their oppenent.
	"""
	Q = {}
	num_empty, empty = get_empty(game)
	#filled_fraction = (boardsize**2-num_empty+1)/boardsize**2
	I1, C1 = resistance(game, empty, color)
	I2, C2 = resistance(game, empty, other(color))

	num_empty, empty = get_empty(game)
	for cell in empty:
		#this makes some sense as an approximation of
		#the conductance of the next game
		C1_prime = C1 + I1[cell]**2/(3*(1-I1[cell]))
		C2_prime = max(0,C2 - I2[cell])

		if(C1_prime>C2_prime):
			Q[cell] = min(1,max(-1,1 - C2_prime/C1_prime))
		elif (C2_prime != 0):
			Q[cell] = min(1,max(-1,C1_prime/C2_prime - 1))
		else:
			Q[cell] = 1.

	output = -1*np.ones((game.size, game.size))
	for cell, value in Q.items():
		output[cell[0]-game.padding, cell[1]-game.padding] = value
	return output