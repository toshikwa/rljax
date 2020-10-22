import os

import haiku as hk
import numpy as np


def save_params(params, path):
    """
    Save parameters.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savez(path, **params)


def load_params(path):
    """
    Load parameters.
    """
    return hk.data_structures.to_immutable_dict(np.load(path))
