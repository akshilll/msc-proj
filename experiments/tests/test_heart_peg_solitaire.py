from experiments.environments.heart_peg_solitaire import heart_peg_state
import pytest
import numpy as np

#########
# TESTING STATE
#########

# Low
def test_state_init():
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
    assert (s.board_directions == [(-1, 0), (1, 0), (0, -1), (0, 1)])
    assert (s.gap_list == [2, 3, 9, 10, 12, 15])

# Low
def test_str():
    # Set the seed
    np.random.seed(2)

    # Random states
    s = heart_peg_state(state = np.random.choice([0, 1], 16))
    s2 = heart_peg_state(state = np.random.choice([0, 1], 16))
    s3 = heart_peg_state(state = np.random.choice([0, 1], 16))

    # Get string representations
    out = s.__str__()
    out2 = s2.__str__()
    out3 = s3.__str__()

    # Test
    assert (out == '[0 1 1 0 0 1 0 1 0 1 0 1 1 1 1 1]')
    assert (out2 == '[1 1 0 0 0 0 1 1 1 0 0 0 1 1 1 0]')
    assert (out3 == '[0 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0]')










def test_eq():
    pass

def test_symmetry_state():
    pass

def test_is_state_legal():
    pass

# Low
def test_is_initial_state():
    pass

# Medium
def test_is_terminal_state():
    pass

def get_available_actions():
    pass

def test_take_action():
    pass

def test_is_action_legal():
    pass 

def test_get_successors():
    pass


###### TESTING ENVIRONMENT

def test_env_reset():
    pass

def test_reset():
    pass

def test_step():
    pass

def test_env_get_available_actions():
    pass