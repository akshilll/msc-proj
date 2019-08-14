import operator
import numpy
import functools
import os
import numpy as np
from barl_simpleoptions.state import State

from abc import ABC, abstractmethod
from typing import List

class heart_peg_state(State):
	'''
	State implementation for peg solitaire on a heart shaped board 
	- inherits from State in barl_simpleoptions

	Args:

	Attributes:
		gap_list (list) : Used to specifify the indices of gaps in the board
		state (list) : List of ones where there are pegs an zeros where there aren't
		symm_state (list) : same as state but for the reflection state
		num_to_coord (list) : list used for mapping indices to coordinates on the board
		symm_num_to_coord (list) : list used for mapping indices of normal state to coordinates of symmetry state
	'''

	def __init__(self, gap_list):	
		
		self.gap_list = gap_list
		self.state = [0 if i in gap_list else 1 for i in range(16)]
		self.symm_state = self.symmetry_state()
		
		# self.num_to_coord = [(-1, 2), (1, 2), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), 
		# 					 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1,-1), (0, -1),
		# 					 (1, -1), (0, -2)]
		# self.symm_num_to_coord = [(-i, j) for (i, j) in self.num_to_coord]
		# self.symm_nums = [self.num_to_coord.index(x) for x in self.symm_num_to_coord]
		
		self.board_directions = [(i, j) for i in range(2) for j in range(2)]
		

	def __str__(self):
		"""String representation of state
		Returns:
			String of state attribute of input
		"""
		return self.state.tostring()


	def __eq__(self, other_state):
	'''Check equality of states 

	Parameters:
		other_state (State): Object for comparison to this State 

	Returns: 
		bool: True iff other_state is equal to state of reflection
	'''
		return (self.symm_state == other_state.state) or (self.state == other_state.state)
	
	
	def symmetry_state(self):
		'''Reflects state in vertical axis
		
		Get symmetry index mapping between state and symm_state
		Set symm_state values as state[symm_index_map]
			
		# Code used to get symm_index_map
		[self.symm_num_to_coord.index(self.num_to_coord[i]) for i in range(len(self.state))]


		Returns:
			out (list) : List which is reflection of input's state attribute
		'''
		# Index mapping from state to symm_state
		symm_index_map = [1, 0, 6, 5, 4, 3, 2, 11, 10, 9, 8, 7, 14, 13, 12, 15]
		out = [self.state[i] for i in symm_index_map]
		return out

	def visualise(self):
		# Print out state attribute separated into rows of board
		
		print(self.state[:2], 
			  self.state[2:7],
			  self.state[7:12],
			  self.state[12:15],
			  self.state[15], sep = "\n")



	
	def is_state_legal(self) -> bool :
		"""
        Returns whether or not the current state is legal.
        Returns:
            bool -- Whether or not this state is legal.
		"""
		# Check board size
		if len(self.state) != 16:
			return False
		
		# The board can't be full of pegs
		if (self.state == np.ones(16)).all():
			return False
		
		# The board can only contain 1s and 0s 
		state_unique = set(self.state)
		if state_unique not in [set([1]), set([0]), set([0, 1])]:
			return False

		# If it's made it this far then it's legal!
		return True

	def is_initial_state(self) -> bool :
		"""
        Returns whether or not this state is an initial state.
        Returns:
            bool -- Whether or not this state is an initial state.
		"""
		start_state = [1] * 16
		start_state[9] = 0
		return self.state == start_state

	# TO DO 
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



	# TO DO
	def get_available_actions(self):
	# where does the state itself come from 
		for i in range(len(self.state)):
			# Look for a gap
			if self.state[i] == 0:
				for (x, y) in self.board_directions:
					pass
		
		return None # List(hashable actions)

	# TO DO
	def take_action(self, action) -> List[State]:

		(gap_number, direction_from) = action
		(x, y) = direction_from 


		return #List(State)
	
	# TO DO
	def is_action_legal(self, action) -> bool :
		# Check that the coordinate is within the board 
		# If the coordinate is on an edge check  that 
		# If there is a gap at the index specified
			# Check the self.gap_list
		# If there are two pegs in the holes in the direction specified
		pass

	def get_successors(self) -> List[State]:
		# Iterate over each action
		available_actions = self.get_available_actions()
		
		# Get list of lists of successor states for each action
		out = [self.take_action(a) for a in available_actions] 
		
		# Flatten list to 1d 
		out = functools.reduce(operator.iconcat, out, [])

		return out
