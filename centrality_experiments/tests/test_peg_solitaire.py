from centrality_experiments.environments.peg_solitaire import peg_solitaire_state, peg_solitaire_env
import pytest
import numpy as np 

#### TESTING STATE ####

def test_state_init():
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    # Terminal losing state
    gap_coord3 = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    # Winning state
    gap_coord4 = [(i, j) for i in range(4) for j in range(4)]
    gap_coord4.remove((1, 2))
    

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state3 = peg_solitaire_state(layout1, gap_coord3)
    state4 = peg_solitaire_state(layout1, gap_coord4)

    # heart states
    gap_coord5 = [(2, 2)]
    gap_coord6 = [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (3, 3), (4, 2)]

    state5 = peg_solitaire_state(layout2, gap_coord5)  
    state6 = peg_solitaire_state(layout2, gap_coord6)

    ## Testing

    assert state1.board_directions == [(-1, 0), (1, 0), (0, -1), (0, 1)]
    assert state2.gap_coords == [(0, 2), (1, 2), (3, 2)]
    assert state3.state == [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    assert state4.grid == [[1] * 4] * 4
    assert state5.layout_path == layout2
    assert state6.initial_state[0][0] == -1
    assert state6.state[1] == [1, 0, 0, 0, 0]


def test_str():
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'    

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)

    assert str(state1) == '[[1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]'
    assert str(state2) == '[[1, 1, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 0, 1]]'

    
def test_eq():

    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    gap_coord5 = [(2, 2)]

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state5 = peg_solitaire_state(layout2, gap_coord5) 
    state6 = peg_solitaire_state(layout2, gap_coord5) 
    state6.layout_path = 'sagasfd'
    
    assert state1 == state1
    assert state2 == state2
    assert state1 != state2
    assert state5 != state1
    assert state6 != state5

def test_get_available_actions():
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    # Terminal losing state
    gap_coord3 = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    # Winning state
    gap_coord4 = [(i, j) for i in range(4) for j in range(4)]
    gap_coord4.remove((1, 2))
    

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state3 = peg_solitaire_state(layout1, gap_coord3)
    state4 = peg_solitaire_state(layout1, gap_coord4)

    # heart states
    gap_coord5 = [(2, 2)]
    gap_coord6 = [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (3, 3), (4, 2)]

    state5 = peg_solitaire_state(layout2, gap_coord5)  
    state6 = peg_solitaire_state(layout2, gap_coord6)

    assert state1.get_available_actions() == [((1, 2), (1, 0)), ((1, 2), (0, -1))]
    assert state2.get_available_actions() == [((0, 2), (0, -1)), ((1, 2), (0, -1)), ((3, 2), (0, -1))]
    assert state3.get_available_actions() == []
    assert state4.get_available_actions() == []
    assert state5.get_available_actions() == [((2, 2), (1, 0)), ((2, 2), (0, -1)), ((2, 2), (0, 1))]
    assert state6.get_available_actions() == [((1, 1), (1, 0)), ((1, 2), (1, 0)), ((2, 0), (0, 1)), ((3, 3), (0, -1)), ((4, 2), (-1, 0))]



def test_take_action():
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    # Terminal losing state
    gap_coord3 = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
   

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state3 = peg_solitaire_state(layout1, gap_coord3)

    # heart states
    gap_coord5 = [(2, 2)]

    state5 = peg_solitaire_state(layout2, gap_coord5)  

    successors1 = state1.take_action(((1, 2), (1, 0)))
    successors2 = state2.take_action(((3, 2), (0, -1)))
    successors5 = state5.take_action(((2, 2), (1, 0)))


    assert len(successors1) == 1
    assert successors1[0].gap_coords == [(2, 2), (3, 2)]
    assert successors2[0].gap_coords == [(0, 2), (1, 2), (3, 1), (3, 0)]
    assert successors5[0].gap_coords == [(3, 2), (4, 2)]

    with pytest.raises(Exception):
        state3.take_action(((3, 2), (0, -1)))
    
    with pytest.raises(Exception):
        state1.take_action(((1, 2), (-1, 0)))


def test_is_terminal_state():
    # Winning
    # Losing
    # non-terminal
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    # Terminal losing state
    gap_coord3 = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    # Winning state
    gap_coord4 = [(i, j) for i in range(4) for j in range(4)]
    gap_coord4.remove((1, 2))
    

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state3 = peg_solitaire_state(layout1, gap_coord3)
    state4 = peg_solitaire_state(layout1, gap_coord4)

    # heart states
    gap_coord5 = [(2, 2)]
    
    state5 = peg_solitaire_state(layout2, gap_coord5)  
    
    assert not state1.is_terminal_state()
    assert not state2.is_terminal_state()
    assert state3.is_terminal_state()
    assert state4.is_terminal_state()
    assert not state5.is_terminal_state()


def test_get_successors():
    layout_dir = './centrality_experiments/environments/peg_solitaire_layouts/'
    layout1 = layout_dir + '4square.txt'
    layout2 = layout_dir + 'heart.txt'

    # Initial state
    gap_coord1 = [(1, 2)]
    gap_coord2 = [(0, 2), (1, 2), (3, 2)]
    # Terminal losing state
    gap_coord3 = [(0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]
    # Winning state
    gap_coord4 = [(i, j) for i in range(4) for j in range(4)]
    gap_coord4.remove((1, 2))
    

    state1 = peg_solitaire_state(layout1, gap_coord1)
    state2 = peg_solitaire_state(layout1, gap_coord2)
    state3 = peg_solitaire_state(layout1, gap_coord3)
    state4 = peg_solitaire_state(layout1, gap_coord4)

    # heart states
    gap_coord5 = [(2, 2)]
    gap_coord6 = [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (3, 3), (4, 2)]

    state5 = peg_solitaire_state(layout2, gap_coord5)  
    state6 = peg_solitaire_state(layout2, gap_coord6)

    successors1 = state1.get_successors()
    successors2 = state2.get_successors()
    successors3 = state3.get_successors()
    successors4 = state4.get_successors()
    successors5 = state5.get_successors()
    successors6 = state6.get_successors()


    assert len(successors1) == 2
    assert len(successors2) == 3
    assert len(successors3) == 0
    assert len(successors4) == 0
    assert len(successors5) == 3
    assert len(successors6) == 5
    

    assert successors1[0].gap_coords == [(2, 2), (3, 2)]
    assert successors2[2].gap_coords == [(0, 2), (1, 2), (3, 1), (3, 0)]
    assert successors5[2].gap_coords == [(2, 3), (2, 4)]
    assert successors6[4].gap_coords == [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (3, 3), (3, 2), (2, 2)]
    

def test_is_state_legal():
    pass


##### TESTING ENVIRONMENT #####

def test_reset():
    pass

def test_step():
    # Win
    # Intermediate
    # Loss
    pass