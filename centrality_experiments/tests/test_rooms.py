from centrality_experiments.environments.rooms import rooms_environment, rooms_state
import pytest
import numpy as np
from copy import deepcopy

layout_dir = "./centrality_experiments/environments/rooms_layouts/"

two_rooms_layout = layout_dir + "two_rooms.txt"
four_rooms_layout = layout_dir + "four_rooms.txt"
six_rooms_layout = layout_dir + "six_rooms.txt"

def test_state_init():
    pos = (2, 2)
    pos2 = (4, 5)
    s = rooms_state(two_rooms_layout, pos)
    s2 = rooms_state(six_rooms_layout, pos2)

    assert s.layout_path == "./centrality_experiments/environments/rooms_layouts/two_rooms.txt"    
    assert s.state == pos
    assert s.action_space == [(-1, 0), (1, 0), (0, -1), (0, 1)]
    assert s.layout_key == {'#': -1, '.': 0, 'S': 1, 'G': 2, 'T': 3}
    
    assert s.grid.shape == (13, 13)
    assert s.grid[2, 2] == 1
    assert np.all(s.grid[0] == np.full(13, -1))
    assert s.grid[9, 10] == 2

    assert s2.grid[22, 16] == 2

def test_str():
    pos = (2, 2)
    pos2 = (4, 5)
    s = rooms_state(two_rooms_layout, pos)
    s2 = rooms_state(six_rooms_layout, pos2)

    assert str(s) == '(2, 2)'
    assert str(s2) == '(4, 5)'


def test_eq():
    pos = (2, 2)
    pos2 = (4, 5)
    s = rooms_state(two_rooms_layout, pos)
    s2 = rooms_state(six_rooms_layout, pos2)

    assert s == s
    assert not s == s2


def test_is_state_legal():

    pos = (2, 2)
    pos2 = (4, 5)
    s = rooms_state(two_rooms_layout, pos)
    s2 = rooms_state(six_rooms_layout, pos2)
    
    assert s.is_state_legal()
    assert s2.is_state_legal()

    s3 = deepcopy(s)
    s3.state = (0,0)    
    assert not s3.is_state_legal()

    s3.state = (0, 10)
    assert not s3.is_state_legal()

    s3.state = (6, 4)
    assert not s3.is_state_legal()




    


