import random

import numpy as np

from src.common.seeding import get_random_state, set_global_seed


def test_get_random_state():
    assert get_random_state() == 42
    assert get_random_state(123) == 123


def test_set_global_seed():
    # Test random
    set_global_seed(42)
    val1 = random.random()
    set_global_seed(42)
    val2 = random.random()
    assert val1 == val2

    # Test numpy
    set_global_seed(42)
    np_val1 = np.random.rand()
    set_global_seed(42)
    np_val2 = np.random.rand()
    assert np_val1 == np_val2
