from experiments.environments.heart_peg_solitaire import heart_peg_state
import pytest
import numpy as np


def test_init():
    '''Testing that objects are instantiated properly and have correct attributes'''
    # Set the seed
    np.random.seed(1)

    # Random state
    state_val = np.random.choice([0, 1], 16)

    # Instantiate an object
    s = heart_peg_state(state = state_val)
    
    # These are the attributes that we care about
    expected_attributes = ['board_directions', 'gap_list', 'get_available_actions', 'get_predecessors',
                            'get_successors', 'get_transition_action', 'is_action_legal', 'is_initial_state',
                            'is_state_legal', 'is_terminal_state', 'num_to_coord', 'state', 'symm_num_to_coord',
                            'symm_state', 'symmetry_state', 'take_action', 'visualise']

    # Test attributes 
    assert (x in dir(s) for x in expected_attributes)

    assert ( s.board_directions == [(-1, 0), (1, 0), (0, -1), (0, 1)] )
    
    assert (s.gap_list == [2, 3, 9, 10, 1])