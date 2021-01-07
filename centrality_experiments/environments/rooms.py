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


# TODO: change to account for stochastic environments.

#String: wall, space, 
# Other attributes: initial state, goal_pos, agent_pos,  

## LAYOUT KEY: {'#': Wall, '.': Space, 'I': Initial, }


class rooms_state(State):
    def __init__(self, layout_path, agent_pos):
        self.layout_path = layout_path
        self.state = agent_pos
        self.action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)] # L, R, U, D
        self.layout_key = {'#': -1, '.': 0, 'S': 1, 'G': 2, 'T': 3}
        self.grid, self.initial_state, self.goal_pos, self.terminal_states = self._process_layout(layout_path, self.layout_key)

    def __str__(self) -> str:
        return str(self.state)
    
    def __eq__(self, other):
        return hash(self.state) == hash(other.state)

    def __hash__(self):
        return hash(str(self))

    def _process_layout(self, layout_path, layout_key):
        layout = np.loadtxt(layout_path, comments="//", dtype=str)

        x_size = len(layout[0])
        y_size = len(layout)
        
        grid = np.zeros([y_size, x_size], dtype=int)
        
        for j, y in enumerate(layout):
            for i, x in enumerate(y):
                grid[j, i] = layout_key[x]

        initial_state = list(zip(*np.where(grid==layout_key['S'])))[0]
        goal_pos = list(zip(*np.where(grid==layout_key['G'])))[0]
        terminal_states = list(zip(*np.where(grid==layout_key['T']))) + [goal_pos]

        return grid, initial_state, goal_pos, terminal_states

    def is_terminal_state(self) -> bool:
        # If the state is in the terminal states.
        return self.state in self.terminal_states

    def is_initial_state(self) -> bool:
        return self.state == self.initial_state

    def is_state_legal(self) -> bool:
        # Is the state within the board - check self.grid
        # Is the state in a gap instead of a wall - check self.grid
        x, y = self.state
        
        if self.grid[y, x] < 0:
            return False
        
        return True

    def get_available_actions(self) -> List:
        available_actions = [a for a in self.action_space if self.is_action_legal(a)]
        return available_actions

    def take_action(self, action) -> List['State']:
        
        x, y = self.state
        action_x, action_y = action

        det_next_pos = (x + action_x, y + action_y)

        det_next_state = rooms_state(self.layout_path, det_next_pos)

        out = []
        if det_next_state.is_state_legal():
            out.append(det_next_state)
        
        candidates = [rooms_state(self.layout_path, (x + i, y + j)) for (i, j) in self.action_space]
        candidates2 = [k for k in candidates if k.is_state_legal() and k not in out]

        out += candidates2

        return out

    def is_action_legal(self, action) -> bool:
        # is action in action space
        if action not in self.action_space:
            return False

        x, y = self.state
        
        # Check for walls in grid
        # Going left when at the leftborder
        if x == 1 and action == (-1, 0):
            return False
        # Going right when at the right border
        if x == self.grid.shape[1] - 2 and action == (1, 0):
            return False

        # Going up when at the top
        if y == 1 and action == (0, -1):
            return False

        # Going down when at the bottom
        if y == self.grid.shape[0] - 2 and action == (0, 1):
            return False

        # If none of the things above have gone wrong then it must be legal
        return True

    def get_successors(self) -> List['State']:
        # Iterate over each action
        available_actions = self.get_available_actions()
		
        # Get list of lists of successor states for each action
        out = [self.take_action(a) for a in available_actions] 
		
        # Flatten list to 1d 
        out = list(set(functools.reduce(operator.iconcat, out, [])))
        
        return out
    
    # TODO: change to return a list for when env is stochastic.
    def get_transition_action(self, next_state : 'State'):
        for a in self.get_available_actions():
            for s_ in self.take_action(a):
                if next_state.state == s_.state:
                    return a
        return (-2, (-2, -2))
    

    def render(self):
        raise(NotImplementedError("Rendering has not been unlocked yet."))


class rooms_environment(BaseEnvironment):
    def __init__(self, options: List['Option'], layout_path, intermediate_reward=-1., win_reward=10., stochasticity_level=1., lose_reward=-10.):
        self.options = options
        self.current_state = None
        self.intermediate_reward = intermediate_reward
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.layout_path = layout_path        
        self.grid, self.initial_state, self.goal_pos, self.terminal_states = self._process_layout(layout_path)
        self.stochasticity_level = stochasticity_level

    def step(self, action) -> Tuple['State', float, bool]:
        poss_successors = self.current_state.take_action(action)
        
        # Deterministic next_state
        if np.random.rand() > self.stochasticity_level: 
            next_state = poss_successors[0]
        else:
            next_state = np.random.choice(poss_successors)
        
        self.current_state = deepcopy(next_state)
        
        terminal = next_state.is_terminal_state()
        if terminal:
            if next_state.state == self.goal_pos:
                reward = self.win_reward
            else:
                reward = self.lose_reward
        else:
            reward = self.intermediate_reward
        
        return deepcopy(next_state), float(reward), terminal

    def reset(self) -> State:
        self.current_state = rooms_state(self.layout_path, self.initial_state)

        return deepcopy(self.current_state)

    def get_available_actions(self) -> List:
        
        if self.current_state is None:
            raise Exception("Current state is None, reset environment")
            
        return self.current_state.get_available_actions()

    def render(self):
        # Construct state and call it's render method
        if self.current_state is None:
            raise Exception("Current state is None, reset environment")
        else:
            self.current_state.render()




