import operator
import numpy
import functools
import os
import math
import json
import random
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from typing import List

class heart_peg_state(State):

	def __init__(self, gap_list):
		"""
	    Instantiate new state
	    Params:
	        - gap_list: indices of holes which are empty. List(Integer)
	     Returns:
	        - State
		"""
		self.gap_list = gap_list
		self.state = np.array([0 if i in gap_list else 1 for i in range(16)])
		self.num_to_coord = [(-1, 2), (1, 2), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), 
							 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1,-1), (0, -1),
							 (1, -1), (0, -2)]
		self.symm_num_to_coord = [(-i, j) for (i, j) in self.num_to_coord]
		self.symm_nums = [self.num_to_coord.index(x) for x in self.symm_num_to_coord]
		self.board_directions = np.array([(i, j) for i in range(2) for j in range(2)])

	def __str__(self):
		return self.state.tostring()


	def __eq__(self, other_state):
	## DO I DO SYMMETRY HERE? YES
		return (self.symmetry_state() == other_state.state()).all() or (self.state() == other_state.state())

	def symmetry_state(self):
		pass

	def is_state_legal(self) -> bool :
		"""
        Returns whether or not the current state is legal.
        Returns:
            bool -- Whether or not this state is legal.
		"""
		
		if len(self.state) != 16:
			return False
		
		if (self.state == np.ones(16)).all():
			return False

		state_unique = set(self.state)

		if not ((state_unique == set([1])) or (state_unique == set([0])) or (state_unique == set([0, 1]))):
			return False
		
		return True

	def is_initial_state(self) -> bool :
		"""
        Returns whether or not this state is an initial state.
        Returns:
            bool -- Whether or not this state is an initial state.
		"""
		start_state = np.ones(16)
		start_state[9] = 0
		return (self.state == start_state).all()


	def is_terminal_state(self) -> bool :
		"""
        Returns whether or not this is a terminal state.
        
        Returns:
			bool -- Whether or not this state is terminal.
		"""
		if np.sum(self.state) == 1:
			return True
		
		for i in range(len(self.state)):
			if self.state[i] == 1:
				pass




	def get_available_actions(self):
	# where does the state itself come from 
		for i in range(len(self.state)):
			# Look for a gap
			if self.state[i] == 0:
				for (x, y) in self.board_directions:
					pass
		
		return None # List(hashable actions)


	def take_action(self, action) -> List[State]:

		(gap_number, direction_from) = action
		(x, y) = direction_from 


		return List(States)

	def is_action_legal(self, action) -> bool :
		# Check that the coordinate is within the board 
		# If the coordinate is on an edge check  that 
		# If there is a gap at the index specified
			# Check the self.gap_list
		# If there are two pegs in the holes in the direction specified
		pass

	def get_successors(self) -> List[State]:
		
		available_actions = self.get_available_actions()
		
		out = [self.take_action(a) for a in available_actions] 
		
		out = functools.reduce(operator.iconcat, out, [])

		return out
