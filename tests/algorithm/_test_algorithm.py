import numpy as np


def _test_algorithm(env, algo):
    state = env.reset()

    # Test step() method.
    _state = algo.step(env, state)
    assert env.observation_space.contains(np.array(_state))

    # Test select_action() method.
    action = algo.select_action(state)
    assert env.action_space.contains(action)

    # Test is_update() method.
    assert isinstance(algo.is_update(), bool)
