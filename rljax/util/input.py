import numpy as np
from gym.spaces import Box


def fake_state(state_space):
    fake_state = state_space.sample()[None, ...]
    if len(state_space.shape) == 1:
        fake_state = fake_state.astype(np.float32)
    return fake_state


def fake_action(action_space):
    if type(action_space) == Box:
        fake_action = action_space.sample().astype(np.float32)[None, ...]
    else:
        fake_action = None
    return fake_action
