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
