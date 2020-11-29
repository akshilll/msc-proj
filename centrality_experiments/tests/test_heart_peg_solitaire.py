from centrality_experiments.environments.heart_peg_solitaire import heart_peg_state, heart_peg_env
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
    assert (out == '[0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]')
    assert (out2 == '[1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0]')
    assert (out3 == '[0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]')

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
    
    # Testing
    assert (s != s2)
    assert (s != s3)
    assert (s == heart_peg_state(state = s.state))
    assert (s5 != s4)


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
    # Set the seed
    np.random.seed(4)

    # Random state
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s2 = heart_peg_state(state=init_state)
    
    # Win state
    end_state = [0] * 16
    end_state[9] = 1
    s3 = heart_peg_state(state=end_state)

    s4 = heart_peg_state(state = [1] * 16)
    s5 = heart_peg_state(state = [0] * 16)
    s6 = heart_peg_state(state = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 2, 0, 1, 2, 1])
    s7 = heart_peg_state(state = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1., 1, 0, 1, 1, 1])

    with pytest.raises(IndexError):
        heart_peg_state(state = [1, 0, 1])

    assert (s.is_state_legal())
    assert (s2.is_state_legal())
    assert (s3.is_state_legal())
    assert (not s4.is_state_legal())
    assert (not s5.is_state_legal())
    assert (not s6.is_state_legal())
    assert (not s7.is_state_legal())

# Low
def test_is_initial_state():
    # Set the seed
    np.random.seed(5)

    # Random state
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s2 = heart_peg_state(state=init_state)
    
    # Win state
    end_state = [0] * 16
    end_state[9] = 1
    s3 = heart_peg_state(state=end_state)

    assert (not s.is_initial_state())
    assert (s2.is_initial_state())
    assert (not s3.is_initial_state())

# Medium
def test_is_terminal_state():
    # Set the seed
    np.random.seed(5)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s2 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s3 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s4 = heart_peg_state(state=init_state)
    
    # End state
    end_state = [0] * 16
    end_state[9] = 1
    s5 = heart_peg_state(state=end_state)

    s6 = heart_peg_state(state=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s7 = heart_peg_state(state=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s8 = heart_peg_state(state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s9 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    s10 = heart_peg_state(state=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s11 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    s12 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1])


    # Testing
    assert (not s.is_terminal_state())
    assert (not s2.is_terminal_state())
    assert (not s3.is_terminal_state())
    assert (not s4.is_terminal_state())
    assert (s5.is_terminal_state())
    assert (s6.is_terminal_state())
    assert (s7.is_terminal_state())
    assert (s8.is_terminal_state())
    assert (s9.is_terminal_state())
    assert (not s10.is_terminal_state())
    assert (not s11.is_terminal_state())
    assert (s12.is_terminal_state())

def test_get_available_actions():
    # Set the seed
    np.random.seed(5)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s2 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    s3 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s4 = heart_peg_state(state=init_state)
    
    # End state
    end_state = [0] * 16
    end_state[9] = 1
    s5 = heart_peg_state(state=end_state)

    s6 = heart_peg_state(state=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s7 = heart_peg_state(state=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s8 = heart_peg_state(state=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s9 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    s10 = heart_peg_state(state=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s11 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    s12 = heart_peg_state(state=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1])

    # Testing
    assert s.get_available_actions() == [(4, (-1, 0)), (8, (0, 1))]
    assert s2.get_available_actions() == [(1, (0, -1)), (6, (-1, 0)), (8, (1, 0)), (8, (0, 1)), (13, (0, 1)), (14, (0, 1))]
    assert s3.get_available_actions() == [(2, (1, 0)), (5, (-1, 0)), (8, (0, 1)), (13, (0, 1))]
    assert s4.get_available_actions() == [(9, (-1, 0)), (9, (1, 0)), (9, (0, -1))]
    assert s5.get_available_actions() == []
    assert s6.get_available_actions() == []
    assert s7.get_available_actions() == []
    assert s8.get_available_actions() == []
    assert s9.get_available_actions() == []
    assert s10.get_available_actions() ==  [(8, (0, 1))]
    assert s11.get_available_actions() == [(9, (-1, 0)), (9, (1, 0))]
    assert s12.get_available_actions() == []


def test_take_action():
    # Set the seed
    np.random.seed(5)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s2 = heart_peg_state(state=init_state)
    
    # End state
    end_state = [0] * 16
    end_state[9] = 1
    s3 = heart_peg_state(state=end_state)
    
    # Left, right, up, down, edge bad, bad state.
    out = s.take_action((4, (-1, 0))) # left
    out2 = s2.take_action((9, (1, 0)))[0] # right
    out3 = out2.take_action((10, (0, 1)))[0] # up
    out4 = s2.take_action((9, (0, -1)))[0] # down

    assert type(out) is list
    assert str(out[0]) == '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]'
    assert str(out2) == '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]'
    assert str(out3) == '[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]'
    assert str(out4) == '[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]'

    with pytest.raises(Exception):
       s3.take_action((10, (-1, 0))) 

def test_is_action_legal():
    # Set the seed
    np.random.seed(5)

    # Random states
    s = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    s3 = heart_peg_state(state=np.random.choice([0, 1], 16).tolist())
    
    end_state = [0] * 16
    end_state[9] = 1
    s3 = heart_peg_state(state=end_state)

    assert s.is_action_legal((4, (-1, 0)))
    assert not s.is_action_legal((4, (1, 1)))
    assert not s.is_action_legal((4, (2, 0)))
    assert not s.is_action_legal((-1, (1, 0)))
    assert not s.is_action_legal((17, (1, 0)))
    

    for i in range(16):
        for j in s.board_directions:
            assert not s3.is_action_legal((i, j))

def test_get_successors():
    for _ in range(100):
        s = heart_peg_state(state = np.random.choice([0, 1], 16))
        succ = s.get_successors()
        assert type(succ) == list
        assert set(succ) == set([s.take_action(a)[0] for a in s.get_available_actions()])


###### TESTING ENVIRONMENT

def test_reset():
    env = heart_peg_env(options = [])
    
    assert env.current_state is None
    assert env.intermediate_reward == 0

    env.reset()

    # Initial state
    init_state = [1] * 16
    init_state[9] = 0
    s = heart_peg_state(state=init_state)

    assert env.current_state == s

def test_step():
    
    # win
    env = heart_peg_env(options = [])
    s = heart_peg_state(state=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    s_ = heart_peg_state(state=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    env.current_state = s
    out = env.step(s.get_available_actions()[0])
    assert out == (s_, env.win_reward, True)

    # Intermediate
    init_state = [1] * 16
    init_state[9] = 0
    s2 = heart_peg_state(state=init_state)
    env2 = heart_peg_env(options = [])
    env2.current_state = s2
    (s2_, r, g) = env2.step(s2.get_available_actions()[0])
    assert r == env2.intermediate_reward
    assert not g
    assert str(s2_) == '[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]'

    # Loss
    env3 = heart_peg_env(options = [])
    s3 = heart_peg_state(state=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    s3_ = heart_peg_state(state=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    env3.current_state = s3
    a = s3.get_available_actions()[0]
    out3 = env3.step(a)
    assert out3 == (s3_, -2., True)





