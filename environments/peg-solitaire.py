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
	'''
	

	def __init__(self, gap_list):	
		
		self.gap_list = gap_list
		self.state = np.array([0 if i in gap_list else 1 for i in range(16)])
		self.symm_state = self.symmetry_state()
		self.num_to_coord = [(-1, 2), (1, 2), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), 
							 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1,-1), (0, -1),
							 (1, -1), (0, -2)]
		self.symm_num_to_coord = [(-i, j) for (i, j) in self.num_to_coord]
		self.symm_nums = [self.num_to_coord.index(x) for x in self.symm_num_to_coord]
		self.board_directions = np.array([(i, j) for i in range(2) for j in range(2)])

	def __str__(self):
		"""String representation of state
		Returns:
			String of state attribute of input
		"""
		return self.state.tostring()


	def __eq__(self, other_state):
	## DO I DO SYMMETRY HERE? YES
		return (self.symm_state == other_state).all() or (self.state == other_state).all()
	def symmetry_state(self):
		'''Reflects state in vertical axis
		Returns:
			out (numpy array) : Array which is reflection of input's state attribute
			
		'''
		out = np.array([self.state[np.where(self.num_to_coord[i] == self.symm_num_to_coord)[0]] for i in range(len(self.state))])
		return out


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

		if state_unique not in [set([1]), set([0]), set([0, 1])]:
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
		
		available_actions = self.get_available_actions()
		
		out = [self.take_action(a) for a in available_actions] 
		
		out = functools.reduce(operator.iconcat, out, [])

		return out
