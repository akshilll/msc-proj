import os
import numpy as np
import functools
import operator
import networkx as nx

from itertools import product
from barl_simpleoptions.state import State
from barl_simpleoptions.environment import Environment, BaseEnvironment
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List, Tuple
import re

class heart_peg_state(State):
	'''State implementation for peg solitaire on a heart shaped board 
		- inherits from State in barl_simpleoptions
	Arguments:
		gap_list (list) -- Indicates indices of holes on board
	
	Attributes:
		gap_list (list) -- Used to specifify the indices of gaps in the board
		state (list) -- List of ones where there are pegs an zeros where there aren't
		symm_state (list) -- same as state but for the reflection state
		num_to_coord (list) -- list used for mapping indices to coordinates on the board
		symm_num_to_coord (list) -- list used for mapping indices of normal state to coordinates of symmetry state
	'''

	def __init__(self, state):	
		self.state = state
		self.gap_list = np.where(np.array(state) == 0)[0].tolist()
		self.symm_state = self.symmetry_state()
		
		self.num_to_coord = [(-1, 2), (1, 2), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1), 
							 (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1,-1), (0, -1),
							 (1, -1), (0, -2)]
		self.symm_num_to_coord = [(-i, j) for (i, j) in self.num_to_coord]
		
		self.board_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
			

	def __str__(self) -> str:
		"""String representation of state
		Returns:
			String of state attribute of input
		"""
		return str(self.state)


	def __eq__(self, other_state) -> bool:
		'''Check equality of states 
		Arguments:
			other_state (State) -- Object for comparison to this State 
		Returns: 
			bool -- True iff other_state is equal to state of reflection
		'''
		# toggle between these lines to account for symmetry or not!
		# return (hash(str(self.state)) == hash(str(other_state.state))) or (hash(str(self.symm_state)) == hash(str(other_state.state)))
		return hash(str(self.state)) == hash(str(other_state.state)) # Symmetry not accounted for
	
	# For symmetry change this to be max([hash(str(state.state), hash(str(symm_state.state)))])
	def __hash__(self):
		return hash(str(self.state))

	def symmetry_state(self):
		'''Reflects state in vertical axis
		
		Get symmetry index mapping between state and symm_state
		Set symm_state values as state[symm_index_map]
			
		# Code used to get symm_index_map
		[self.symm_num_to_coord.index(self.num_to_coord[i]) for i in range(len(self.state))]
		Returns:
			out (list) -- List which is reflection of input's state attribute
		'''
		# Index mapping from state to symm_state
		symm_index_map = [1, 0, 6, 5, 4, 3, 2, 11, 10, 9, 8, 7, 14, 13, 12, 15]
		out = [self.state[i] for i in symm_index_map]
		return out

	def visualise(self) -> None:
		'''Print out state attribute separated into rows of board'''
		
		print('    ', self.state[0], '   ', self.state[1], "\n", 
			  self.state[2:7], "\n",
			  self.state[7:12], "\n", 
			  '  ', self.state[12:15], "\n",
			  '      ', self.state[15])


	def is_state_legal(self) -> bool:
		"""
        Determines whether or not the current state is legal.
        Returns:
            bool -- Whether or not this state is legal.
		"""
		# Check board size
		if len(self.state) != 16:
			return False
		
		# The board can't be full of pegs
		if self.state == [1] * 16:
			return False
		
		# It can't be empty either
		if self.state == [0] * 16:
			return False
		
		# Check that the state has only integers in it 
		if not all(isinstance(n, int) for n in self.state): 
			return False

		# The board can only contain 1s and 0s 
		state_unique = set(self.state)
		if state_unique != set([0, 1]):
			return False

		# If it's made it this far then it's legal!
		return True

	def is_initial_state(self) -> bool:
		"""
        Returns whether or not this state is an initial state.
        Returns:
            bool -- Whether or not this state is an initial state.
		"""
		# Ideal initial state
		start_state = [1] * 16
		start_state[9] = 0

		return self.state == start_state

	def is_terminal_state(self) -> bool:
		"""
        Returns whether or not this is a terminal state.
        
        Returns:
			bool -- Whether or not this state is terminal.
		"""
		# If there's only one peg left
		if np.sum(self.state) == 1:
			return True
		
		# If there are no available actions left
		if len(self.get_available_actions()) == 0:
			return True
		
			
		# Must not be terminal if it's got this far
		return False

	def get_available_actions(self) -> List:
		""" Gets available actions for state by iterating to find holes and checking if there are two pegs next to
		
		Returns -- List[(gap_index, direction_of_peg_from_gap (x,y))]
		"""

		# Iterate over all gaps and board directions and only get legal actions
		out = [a for a in product(self.gap_list, self.board_directions) if self.is_action_legal(a)]

		return out

	
	def take_action(self, action) -> List['State']:
		''' Returns possible successors - only one in this case as env is deterministic
		
		Arguments:
			action (int, (int, int)) -- action to be taken in gap_index, direction format
		Returns:
			List[State] -- List of possible successor states (in this case only one due to deterministic env)
		'''
		# Check action is legal
		if not self.is_action_legal(action):
			raise Exception('Action is not legal')
		
		# Unpack action
		(gap_idx, direction_from) = action
		(x, y) = direction_from 
		
		# Alter current state to get new state
		new_state = deepcopy(self.state)
		
		# Get coord of gap
		(gap_x, gap_y) = self.num_to_coord[gap_idx]

		# Find index of coord of new gap
		new_gap_idx = self.num_to_coord.index((gap_x + x, gap_y + y))
		new_gap_idx2 = self.num_to_coord.index((gap_x + 2*x, gap_y + 2*y))
		
		# Alter current state to make new state
		new_state[gap_idx] = 1
		new_state[new_gap_idx] = 0
		new_state[new_gap_idx2] = 0
		s = heart_peg_state(state = new_state)
		
		return [s]

	
	def is_action_legal(self, action) -> bool:
		''' Check action is legal
		Check gap is on board
		Check direction is legal
		If gap is on edge, only certain directions are allowed
		If there is only 1 peg then False
		Arguments:
			action (int, (int, int)): gap number, direction from
		
		'''
		
		# If there are two pegs in the holes in the direction specified
		(gap_idx, direction_from) = action
		(x, y) = direction_from 

		# Check the gap is on the board
		if gap_idx not in range(16):
			return False
		
		# Check there is a gap at the specified index
		if self.state[gap_idx] != 0:
			return False
		
		# Check the direction is legal
		if direction_from not in self.board_directions:
			return False
		
		# Check there is more than one peg
		if np.sum(self.state) == 1:
			return False	
		
		# Get coordinates of relevant slots
		(gap_coord_x, gap_coord_y) = self.num_to_coord[gap_idx]
		peg_jumping_coord = (gap_coord_x + 2*x, gap_coord_y + 2*y)
		peg_middle_coord = (gap_coord_x + x, gap_coord_y + y)

		# Check that 1 slot away in the direction still on the board
		if peg_middle_coord not in self.num_to_coord: 
			return False

		# Check that 2 slots away in the direction still on the board
		if peg_jumping_coord not in self.num_to_coord:
			return False
		
		# Get indices of relevant slots to check if there are pegs in them
		peg_jumping_idx = self.num_to_coord.index(peg_jumping_coord)
		peg_middle_idx = self.num_to_coord.index(peg_middle_coord)
		
		# Check that there is a peg 2 slots away from gap
		if self.state[peg_jumping_idx] != 1:
			return False
		
		# Check that there is a peg 1 slot away from gap
		if self.state[peg_middle_idx] !=1:
			return False
		
		# If it has gotten this far it must be legal!
		return True

	def get_successors(self) -> List['State']:
		# Iterate over each action
		available_actions = self.get_available_actions()
		
		# Get list of lists of successor states for each action
		out = [self.take_action(a) for a in available_actions] 
		
		# Flatten list to 1d 
		out = functools.reduce(operator.iconcat, out, [])

		return out
	
	def get_transition_action(self, next_state : 'State') :
		"""
        Get list of actions to enable transition
        Arguments:
            next_state (state) -- successive state to work out action for
        Returns:
            List[actions] -- List of actions 
        """
		out = [a for a in self.get_available_actions() if next_state in self.take_action(a)]
		return out



class heart_peg_env(BaseEnvironment):
	"""Implementation of an environment for the heart shaped version of peg solitaire
	Attributes:
		options List[Option] -- List of Options found for this environment
		current_state heart_peg_state -- The current state of the environment
		intermediate_reward float -- Intermediate reward for acting in the environment
		win_reward float -- Reward for winning the game
	"""
	
	def __init__(self, options : List['Option'], intermediate_reward=0., win_reward=1.):
		
		self.options = options
		self.current_state = None
		self.intermediate_reward = intermediate_reward
		self.win_reward = win_reward		

	def step(self, action) -> Tuple['State', float, bool]:
		"""
        One step transitions of environment
        
        Arguments:
            action -- Action to be taken by the agent
        Returns:
            next_state (State) -- Successive state after the action is taken
            reward (float) -- Immediate reward
            game_over (bool) -- If the game is over after this step
        next_state, reward, game_over
        """
		# Get next state as first elem because deterministic
		self.current_state = deepcopy(self.current_state.take_action(action)[0])
		
		# Count number of pegs remaining to get reward
		pegs_remaining = np.sum(self.current_state.state)
		

		# If the game is over
		if self.current_state.is_terminal_state():
			game_over = True
			# Check if the game is won
			if pegs_remaining == 1:
				reward = self.win_reward

			else:
				reward = -pegs_remaining
				
		# The game is still afoot!
		else:
			
			game_over = False
			reward = self.intermediate_reward

		return deepcopy(self.current_state), float(reward), game_over
		

	def reset(self):
		""" Resets environment to initial state
		Returns : 
		state -- initial state of the board
		"""
		# Construct start state
		start_state = [1] * 16
		start_state[9] = 0

		# Assign to current_state
		self.current_state = heart_peg_state(state = start_state)
		
		return deepcopy(self.current_state)


	def get_available_actions(self):
		""" Wrapper function for state method of same name
		Returns:
		List[action] -- List of all available actions in the current state of the environment
		"""
		if self.current_state is None:
			raise Exception("Current state is None, reset environment")
		
		return self.current_state.get_available_actions()
	
	# Improved version of Josh's code
	# def get_available_options(self):
	# 	"""Finds all available options for this 

	# 	Returns : 
	# 	List[Option] -- list of options whose initiation set contain the current state
	# 	"""
	# 	available_options = [option for option in self.options if option.initiation(self.current_state)]
	# 	return available_options

def string_to_hps(string):
    """ Converts string from node in graph to state

    Arguments:
    string -- String containing node state

    Returns:
    hps -- State object corresponding to string
    """
    # Format the string to remove everything but the numbers
    new_string = re.sub("[\[\], ]+", "", string)
    
    # Convert to list of integers
    state = [int(i) for i in new_string]
    
    # Construct state
    hps = heart_peg_state(state=state)

    assert hps.is_state_legal(),  str(hps)

    return state