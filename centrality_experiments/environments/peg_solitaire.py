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
    
    def __init__(self, layout_path, gap_coords):
        self.layout_path = layout_path
        self.layout_key = {'#': -1, '0': 0, '1': 1}
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.grid, self.initial_state = self._process_layout(layout_path, self.layout_key)    
        self.gap_coords = gap_coords
        self.state = self._generate_state(self.grid, self.gap_coords)
        
        

    def __str__(self) -> str:
        """String representation of state
        Returns:
        String of state attribute of input
		"""
        return str(self.state)

    def __eq__(self, other_state) -> bool:
        """
        :param other_state: (State) -- Object for comparison to this State 
        :returns: (bool) -- True iff other_state is equal to state of reflection
        """
        if self.layout_path == other_state.layout_path:
            if hash(str(self)) == hash(str(other_state)):  # Symmetry not accounted for
                return True
	
	    # For symmetry change this to be max([hash(str(state.state), hash(str(symm_state.state)))])

    def __hash__(self):
        return hash(str(self.state))
        
    def _process_layout(self, layout_path, layout_key):
        # return with dtype as int      
        layout = np.loadtxt(layout_path, comments="//", dtype=str)

        x_size = len(layout[0])
        y_size = len(layout)
        
        initial_state = np.zeros([y_size, x_size], dtype=int)
        
        for j, y in enumerate(layout):
            for i, x in enumerate(y):
                initial_state[j, i] = layout_key[x]

        return initial_state
    
    def is_initial_state(self) -> bool:
        """Check the grid for the string S
        """
        return self.state == self.initial_state

    def _generate_state(self, grid, gap_coords):
        state = deepcopy(self.grid)
        init_gap = state[state==0] = 1

        for x,y in gap_coords:
            state[x, y] = 0

        return state

    def get_available_actions(self):
        available_actions = [a for a in self.action_space if self.is_action_legal(a)]
        return available_actions

    # TODO: finish
    def take_action(self, action) -> List['State']:
        
        # Check action is legal
        if not self.is_action_legal(action):
            raise Exception('Action is not legal')

        (gap_coord, direction_from) = action 

        gap_x, gap_y = gap_coord
        direction_x, direction_y = direction_from

        gap_coords = deepcopy(self.gap_coords)
        
        peg_mid_coord = gap_x + direction_x, gap_y + direction_y
        peg_end_coord = gap_x + 2*direction_x, gap_y + 2*direction_y

        # Delete former gap_coord from list
        # Add new gap_coords to list
        # Generate state 
        gap_coords = deepcopy(self.gap_coords)
        gap_coords.remove(gap_coords)
        gap_coords = gap_coords + [peg_mid_coord, peg_end_coord]

        layout_path = self.layout_path

        new_state = peg_solitaire_state(layout_path, gap_coords)

        return [new_state]





    def is_terminal_state(self) -> bool:
        if len(self.get_available_actions()) == 0:
            return True
        
        return False
    
    def get_successors(self) -> List['State']:
        # Iterate over each action
        available_actions = self.get_available_actions()
		
		# Get list of lists of successor states for each action
        out = [self.take_action(a) for a in available_actions] 
		
		# Flatten list to 1d 
        out = functools.reduce(operator.iconcat, out, [])
        
        return out

    def get_transition_action(self, next_state : 'State') -> List:
        out = [a for a in self.get_available_actions() if next_state in self.take_action(a)]
        return out

    def is_state_legal(self) -> bool:
        # Right shape and type
        # Not too many pegs
        # Not empty
        # Not just -1s
        # only ints
        # Check there is only 1s, 0s and -1s
        if self.grid.shape != self.state.shape:
            return False

        if self.state.dtype != np.dtype(np.int32):
            return False
        
        if 0 not in self.state:
            return False
        
        if 1 not in self.state:
            return False
        
        return True

    def is_action_legal(self, action) -> bool:
        # Check it's in action space
        # Check there is a gap and it is on the board
        # Check the two pegs indices are one the board.
        # Check there are two pegs in the right direction

        (gap_coord, direction_from) = action 

        gap_x, gap_y = gap_coord
        direction_x, direction_y = direction_from

        state_shape_x, state_shape_y = self.state.shape
        if direction_from not in self.action_space:
            return False

        if gap_coord not in self.gap_coords:
            return False
        
        peg_mid_x, peg_mid_y = gap_x + direction_x, gap_y + direction_y
        peg_end_x, peg_end_y = gap_x + 2*direction_x, gap_y + 2*direction_y

        # Check old peg slots (new gap slots) are on the board
        if peg_mid_x >= state_shape_x or peg_end_x >= state_shape_x or peg_mid_y >= state_shape_y or peg_end_y >= state_shape_y:
            return False
        
        if self.state[peg_mid_x, peg_mid_y] != 1:
            return False
        
        if self.state[peg_end_x, peg_end_y] != 1:
            return False
        
        return True

class peg_solitaire_env(BaseEnvironment):
    def __init__(self, options):
        self.options = options
        self.current_state = current_state

    def reset(self) -> State:
        pass

    def step(action) -> State:
        pass