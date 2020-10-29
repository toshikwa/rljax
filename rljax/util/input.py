import numpy as np
from gym.spaces import Box


def fake_state(state_space):
    state = state_space.sample()[None, ...]
    if len(state_space.shape) == 1:
        state = state.astype(np.float32)
    return state


def fake_action(action_space):
    if type(action_space) == Box:
        action = action_space.sample().astype(np.float32)[None, ...]
    else:
        NotImplementedError
    return action
