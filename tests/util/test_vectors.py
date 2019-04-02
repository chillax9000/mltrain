from util import vector
import numpy as np

def test_one_hot():
    assert np.all(vector.one_hot(7, 2) == np.array([0, 0, 1, 0, 0, 0, 0]))
