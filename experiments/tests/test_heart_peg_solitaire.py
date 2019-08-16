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
    state_val = np.random.choice([0, 1], 16).tolist()

    # Instantiate an object
    s = heart_peg_state(state=state_val)
    
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
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s2 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s3 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())

    # Get string representations
    out = s.__str__()
    out2 = s2.__str__()
    out3 = s3.__str__()

    # Test
    assert (out == '[0 1 1 0 0 1 0 1 0 1 0 1 1 1 1 1]')
    assert (out2 == '[1 1 0 0 0 0 1 1 1 0 0 0 1 1 1 0]')
    assert (out3 == '[0 1 0 0 1 1 1 0 0 0 0 1 1 1 1 0]')

def test_eq():
    # Set the seed
    np.random.seed(3)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s2 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s3 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    init_state = [1] * 16
    init_state[9] = 0
    s4 = heart_peg_state(state=init_state)
    
    end_state = [0] * 16
    end_state[9] = 1
    s5 = heart_peg_state(state=end_state)

    s_symm = heart_peg_state(state=s.symm_state)
    s2_symm = heart_peg_state(state=s2.symm_state)
    s3_symm = heart_peg_state(state=s3.symm_state)

    # Testing
    assert (s != s2)
    assert (s != s3)
    assert (s == heart_peg_state(state = s.state))
    assert (s5 != s4)

    # Symmetry
    assert (s == s_symm)
    assert (s2 == s2_symm)
    assert (s3 == s3_symm)



def test_symmetry_state():
    # Set the seed
    np.random.seed(3)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    init_state = [1] * 16
    init_state[9] = 0
    s2 = heart_peg_state(state=init_state)
    
    end_state = [0] * 16
    end_state[9] = 1
    s3 = heart_peg_state(state=end_state)

    s_symm = heart_peg_state(state=s.symm_state)
    
    # Testing
    assert (s.symm_state == [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    assert (s.symm_state == s_symm.state)
    assert (s2.symm_state == [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    assert (s3.symm_state == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    assert (s2.symm_state == s2.state)
    assert (s.state == s_symm.symm_state)






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