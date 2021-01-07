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


class peg_solitaire_state(State):
    """State representation of any peg solitaire board"""
    def __init__(self, layout_path, gap_coords):
        """Generate the state and useful attributes.
        Args:
            layout_path (str): Path to the file which dictates the board layout
            gap_coords (list): List containing tuples of coordinates of gaps on the current board. This specifies the state.
        """
        self.layout_path = layout_path
        # Keep a dict to translate the txt document into a numpy array
        self.layout_key = {'#': -1, '0': 0, '1': 1} 

        # Constant action space 
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Generate a template array of the board and init state for future use
        self.grid, self.initial_state = self._process_layout(layout_path, self.layout_key)    
        self.gap_coords = gap_coords
        
        self.state = self._generate_state(self.grid, self.gap_coords)

    def __str__(self) -> str:
        """String representation of state
        Returns:
            string: string of state attribute of object
		"""
        return str(self.state)

    def __eq__(self, other_state) -> bool:
        """
        Args:
            other_state: (State): Object for comparison to this State 
        
        Returns:
            bool: True iff the hash of other_state is equal to the hash of this state
        """
        if self.layout_path == other_state.layout_path:
            if hash(str(self)) == hash(str(other_state)): 
                return True

        # For symmetry change this to be max([hash(str(state.state), hash(str(symm_state.state)))])
        return False
	    

    def __hash__(self):
        """Hash functionality for state

        Returns:
            int: Integer representation of the state
        """
        return hash(str(self.state))
        
    def _process_layout(self, layout_path, layout_key):
        """Given a string MDP and key, return a domain representation

        Args:
            layout_path (string): Path to txt file describing environment
            layout_key (dict): Dictates substitution of strings to numbers for domain grid

        Returns:
            numpy array: Grid with -1s as invalid spots and 1s as valid spots specifying the template board
            numpy array: Initial state with -1s as invalid spots, 1s as valid spots and 0s as empty spots
        """
        # load an array of strings specifying the board shape and 
        layout = np.loadtxt(layout_path, comments="//", dtype=str)

        x_size = len(layout[0])
        y_size = len(layout)
        

        # Create an array of -1s as invalid spots, 0s as gaps and 1s as filled spots
        initial_state = np.zeros([y_size, x_size], dtype=int)
        for j, y in enumerate(layout):
            for i, x in enumerate(y):
                initial_state[j, i] = layout_key[x]
        
        # Set all valid spots on board to 1 for template
        grid = deepcopy(initial_state)
        grid[initial_state==0] = 1

        return grid, initial_state
    
    def is_initial_state(self) -> bool:
        """Check the grid for the string S

        Returns: 
            bool: True iff the state is the initial state.
        """
        return self.state == self.initial_state

    def _generate_state(self, grid, gap_coords):
        """Generate the state array from a template and specification of gaps

        Args:
            grid (numpy array): Template layout array of the peg solitaire board
            gap_coords (list): Contains tuple coordinates of gaps on the board

        Returns:
            numpy array: State array with -1s denoting invalid spots, 0s denoting empty valid spots and 1s denoting filled spots on the board
        """
        state = deepcopy(self.grid)
        
        for x, y in gap_coords:
            state[x, y] = 0

        return state

    # TODO: Use filter instead of list comp
    def get_available_actions(self):
        """Wrapper function which returns legal actions from action space

        Returns:
            List: Available actions in current state
        """
        available_actions = [a for a in self.action_space if self.is_action_legal(a)]
        return available_actions

    def take_action(self, action) -> List['State']:
        """Given an action, performs the transition to return a list of possible successor states.

        Raises:
            Exception: If the action is not legal in the current state

        Returns:
            list: List of possible next states.
        """
        # Check action is legal
        if not self.is_action_legal(action):
            raise Exception('Action is not legal.')

        (gap_coord, direction_from) = action 

        gap_x, gap_y = gap_coord
        direction_x, direction_y = direction_from
        
        # Find new gap coordinates
        peg_mid_coord = gap_x + direction_x, gap_y + direction_y
        peg_end_coord = gap_x + 2*direction_x, gap_y + 2*direction_y

        # Delete former gap_coord from list
        gap_coords = deepcopy(self.gap_coords)
        gap_coords.remove(gap_coord)

        # Add new gap_coords to list
        gap_coords = gap_coords + [peg_mid_coord, peg_end_coord]

        # Generate state 
        layout_path = self.layout_path
        new_state = peg_solitaire_state(layout_path, gap_coords)

        return [new_state]
        
    def is_terminal_state(self) -> bool:
        """Determines if the current state is terminal

        Returns:
            bool: True iff there are no available legal actions (the game is thus over).
        """
        return len(self.get_available_actions()) == 0

    def get_successors(self) -> List['State']:
        """Get a list of possible next states from this state

        Returns:
            list: Possible next states from this state
        """

        available_actions = self.get_available_actions()
		
		# Get list of lists of successor states for each action
        out = [self.take_action(a) for a in available_actions] 
		
		# Flatten list to 1d 
        out = functools.reduce(operator.iconcat, out, [])
        
        return out

    def get_transition_action(self, next_state : 'State') -> List:
        """Determine the possible actions from a state, next_state pair - exhaustively iterates over available_actions

        Args:
            next_state (State): The state after the transition

        Returns:
            List: Possible actions which can cause the state transition
        """
        out = [a for a in self.get_available_actions() if next_state in self.take_action(a)]
        return out

    def is_state_legal(self) -> bool:
        """Check if the state is valid based on a few test cases

        Returns:
            bool: True iff the board is valid
        """
        
        # Check the board is the right shape
        if self.grid.shape != self.state.shape:
            return False

        # Check the board only has ints
        if self.state.dtype != np.dtype(np.int32):
            return False
        
        # Completely full board which is not possible
        if 0 not in self.state:
            return False
        
        # Completely empty board which is not possible
        if 1 not in self.state:
            return False
        
        return True

    def is_action_legal(self, action) -> bool:
        """Determines an actions legality in the current state if direction, gap_coord and the combination of both are legal

        Args:
            action (Action): Action to test

        Returns:
            bool: True iff the action is legal. 
        """
        (gap_coord, direction_from) = action 

        gap_x, gap_y = gap_coord
        direction_x, direction_y = direction_from

        state_shape_x, state_shape_y = self.state.shape

        # Is the action in the action space?
        if direction_from not in self.action_space:
            return False

        # Is the initial gap present in the board?
        if gap_coord not in self.gap_coords:
            return False
        
        # Check old peg slots (new gap slots) are on the board
        peg_mid_x, peg_mid_y = gap_x + direction_x, gap_y + direction_y
        peg_end_x, peg_end_y = gap_x + 2*direction_x, gap_y + 2*direction_y

        if peg_mid_x >= state_shape_x or peg_end_x >= state_shape_x or peg_mid_y >= state_shape_y or peg_end_y >= state_shape_y:
            return False
        if self.state[peg_mid_x, peg_mid_y] != 1:
            return False
        if self.state[peg_end_x, peg_end_y] != 1:
            return False
        
        # Must be legal at this point.
        return True

class peg_solitaire_env(BaseEnvironment):
    """Implementation of an environment for the peg solitaire game"""

    def __init__(self, options: list, layout_path: str, init_gap_coords: list, win_reward: float):
        self.options = options
        self.current_state = None
        self.initial_gaps = init_gap_coords
        self.layout_path = layout_path
        self.win_reward = win_reward

    def reset(self) -> State:
        """Resets the board to initial state and generates the state

        Returns:
            State: Initial state
        """ 
        self.current_state =  peg_solitaire_state(self.layout_path, self.initial_gaps)
        return deepcopy(self.current_state)

    def step(self, action) -> Tuple['State', float, bool]:
        """Executes one step transitions

        Args:
            action (Action): Action to be executed

        Returns:
            State: Next state after transition
            float: One step reward of transition
            bool: True iff the next state is terminal
        """
        # Get next state
        self.current_state = deepcopy(self.current_state.take_action(action)[0])  # Change for stochastic transitions
        
        # Check if terminal
        if self.current_state.is_terminal_state():
            game_over = True
            # If there is only one peg left
            num_pegs = np.sum(self.current_state.state == 1)
            # Winning terminal reward
            if num_pegs == 1:
                reward = self.win_reward
            # Losing terminal reward
            else:
                reward = -num_pegs
        
        # Non-terminal transition
        else:
            game_over = False
            reward = 0.
        
        return deepcopy(self.current_state), reward, game_over         

    def get_available_actions(self) -> List:
        """Wrapper function for state.get_available_actions

        Raises:
            Exception: If the environment has not been reset yet - there is no current state

        Returns:
            list: List of available actions in current state.
        """
        if self.current_state is None:
            raise Exception("No current state to call available actions - reset the environment")
        
        available_actions =  self.current_state.get_available_actions()
        return available_actions