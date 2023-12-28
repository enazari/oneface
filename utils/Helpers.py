import numpy as np


def l2norm(vec):
    return vec / np.sqrt(np.sum(vec ** 2, -1, keepdims=True))
