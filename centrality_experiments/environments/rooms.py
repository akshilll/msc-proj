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

#String: wall, space, 
# Other attributes: initial state, goal_pos, agent_pos,  


class rooms_state(State):
    def __init__(self, layout_path, agent_pos):
        self.layout_path = layout_path
        self.initial_state, self.grid, self.terminal_state, self.goal_pos, self.initial_agent_pos = self._process_layout(layout_path)
        self.terminal_state = 
        self.action_space = {'L': (-1, 0), 'R': (1, 0), 'U': (0, -1), 'D': (0, 1)}


    def __str__(self) -> str:
        return str(self.state.tolist())

    def __repr__(self):
        pass
    
    def __eq__(self, other):
        return hash(self.state) == hash(other.state)

    def __hash__(self):
        return hash(str(self))

    def is_terminal_state(self) -> bool:
        # If the state is in the terminal states.
        return self.state == self.terminal_state 

    def is_initial_state(self) -> bool:
        pass

    def get_available_actions(self) -> List:
        pass

    def take_action(self, action) -> List['State']:
        pass

    def is_action_legal(self, action) -> bool:
        pass
    
    def get_successors(self) -> List['State']:
        available_actions = self.get_available_actions()

    def get_transition_action(self, next_state : 'State') :
        pass
    
    def _process_layout(self, layout_path):
        return grid, goal_pos, initial_agent_pos

    def _generate_state(self, grid, pos):
        pass

    def render(self):
        pass


class rooms_environment(BaseEnvironment):
    def __init__(self, options: List['Option'], layout_path, intermediate_reward=-1., win_reward=10, stochasticity=1.):
        self.options = options
        self.current_state = None
        self.intermediate_reward = intermediate_reward
        self.win_reward = win_reward
        self.layout_path = layout_path 
        self.str_mdp = np.loadtxt(layout_path)

    def step(self, action) -> Tuple['State', float, bool]:
        pass

    def reset(self) -> State:
        pass

    def get_available_actions(self) -> List:
        return self.current_state.get_available_actions()

    def render(self):
        # Construct state and call it's render method
        pass

    def _process_layout(self, layout_path):
        return grid, goal_pos, initial_agent_pos





