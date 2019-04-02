import numpy as np


def one_hot(dim: int, index: int):
    return np.eye(dim)[index]
